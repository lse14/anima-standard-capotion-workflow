"""Core-owned bounded validation barrier for Export.

Directory staging and commit deliberately remain outside this runner until the
whole journal protocol is available.  This runner never mutates a dataset.
"""
from __future__ import annotations

import hashlib, json, uuid
from typing import Protocol

from .contracts import CURRENT_JOB_CONFIG_SCHEMA_VERSION, MODULE_BATCH_SIZE_BOUNDS, SampleIssue, WorkLease, sha256_json
from .db import MAX_PAGE_SIZE, StateDatabase
from .export_summary import CONVERSION_CODES, CONVERTED_SAMPLES_CODE
from .overlay import WorkingAnnotationView
from .path_safety import safe_relative_path, sha256_file
from .scheduler import BoundedScheduler, SchedulerError
from .stdio_transport import StdioJsonlTransportError
from .token_budget_overlay import TokenBudgetOverlayError, TokenBudgetOverlayWriter
from .worker_protocol import ProtocolEnvelopeV1, ProtocolError

RUNTIME_ID="export"; OWNER="export"
_CLASSIFY_REPAIR_CODES = frozenset({"json_missing_or_blank", "count_invalid", "cross_field_tag_collision", "payload_all_empty", "tag_not_flat_txt_representable"})
_NL_REPAIR_CODES = frozenset({"nl_control_character", "nl_too_large"})
_REPAIR_ORDER = {"classify": 0, "nl": 1}

class ExportTransport(Protocol):
    def exchange(self, request: ProtocolEnvelopeV1) -> ProtocolEnvelopeV1: ...

class ExportRunnerError(RuntimeError): pass

class ExportRunner:
    def __init__(self,database:StateDatabase,scheduler:BoundedScheduler,transport:ExportTransport,view:WorkingAnnotationView,*,job_id:str,worker_instance_id:str)->None:
        self.database,self.scheduler,self.transport,self.view=database,scheduler,transport,view; self.job_id,self.worker_instance_id=job_id,worker_instance_id; self._hello=False
    def _config(self)->tuple[str,dict[str,object]]:
        job=self.database.get_job(self.job_id); config=json.loads(str(job["config_json"]))
        schema_version=config.get("schemaVersion") if isinstance(config,dict) else None
        caption_format = config.get("captionFormat") if isinstance(config, dict) else None
        batches = config.get("moduleBatchSize") if isinstance(config, dict) else None
        if (
            not isinstance(config, dict)
            or sha256_json(config) != job["config_hash"]
            or schema_version != CURRENT_JOB_CONFIG_SCHEMA_VERSION
            or int(job["config_schema_version"]) != CURRENT_JOB_CONFIG_SCHEMA_VERSION
            or "profile" in config
            or not isinstance(batches, dict)
            or set(batches) != set(MODULE_BATCH_SIZE_BOUNDS)
            or any(type(batches[key]) is not int or not MODULE_BATCH_SIZE_BOUNDS[key][0] <= batches[key] <= MODULE_BATCH_SIZE_BOUNDS[key][1] for key in MODULE_BATCH_SIZE_BOUNDS)
            or not isinstance(caption_format, dict)
            or set(caption_format) != {"replaceUnderscoresWithSpaces", "preserveEscapes", "triggersEnabled", "triggerTerms", "flatTxtLayout"}
            or caption_format.get("flatTxtLayout") not in {"single_line", "nl_newline"}
            or config.get("export",{}).get("format") not in {"json","flat_txt","both"}
        ): raise ExportRunnerError("frozen export configuration is invalid")
        return str(job["config_hash"]),config
    def _exchange(self,method:str,payload:dict[str,object],config_hash:str)->dict[str,object]:
        request=ProtocolEnvelopeV1("1.0","request",f"export-{uuid.uuid4().hex}",RUNTIME_ID,OWNER,method,payload,jobId=self.job_id,configHash=config_hash)
        try: response=self.transport.exchange(request)
        except StdioJsonlTransportError: raise
        except Exception as exc: raise ExportRunnerError("export transport failed") from exc
        if not isinstance(response,ProtocolEnvelopeV1) or response.kind!="response" or response.replyTo!=request.messageId or response.runtimeId!=RUNTIME_ID or response.owner!=OWNER or response.jobId!=self.job_id or response.configHash!=config_hash: raise ExportRunnerError("export response identity mismatch")
        if response.method=="error": raise ExportRunnerError("export worker rejected request")
        if response.method != ("hello" if method=="hello" else "result"): raise ExportRunnerError("export response method mismatch")
        return response.payload
    def _repair(self,field_errors:list[dict[str,object]],config:dict[str,object])->tuple[bool,str|None]:
        classify = config.get("classify")
        nl = config.get("nl")
        classify_enabled = isinstance(classify, dict) and classify.get("enabled") is True
        nl_enabled = isinstance(nl, dict) and nl.get("enabled") is True
        repairs: list[str] = []
        for item in field_errors:
            code = item.get("code")
            if code in _CLASSIFY_REPAIR_CODES and classify_enabled:
                repairs.append("classify")
            elif code in _NL_REPAIR_CODES and nl_enabled:
                repairs.append("nl")
            else:
                # Syntax, encoding, type and configuration errors cannot be
                # made safe by resending the frozen task to a worker.
                return False, None
        return True, min(repairs, key=_REPAIR_ORDER.__getitem__) if repairs else None
    def _issue(self,lease:WorkLease,row:object,outcome:dict[str,object],config:dict[str,object])->None:
        errors=outcome.get("fieldErrors")
        if not isinstance(errors,list) or not errors or not all(isinstance(item,dict) and isinstance(item.get("code"),str) and (item.get("field") is None or isinstance(item.get("field"),str)) for item in errors): raise ExportRunnerError("export field errors are invalid")
        retriable,repair=self._repair(errors,config)
        self.scheduler.skip_export_with_issue(lease,SampleIssue(issueId=hashlib.sha256(f"{self.job_id}\0{lease.sampleId}\0export\0final_json_invalid".encode()).hexdigest(),jobId=self.job_id,sampleId=lease.sampleId,relativeImagePath=str(row["relative_image_path"]),moduleId="export",code="final_json_invalid",severity="error",blocking=True,retriable=retriable,message="final JSON validation failed",attempt=lease.attempt,repairStartModule=repair,fieldErrors=tuple({key:value for key,value in item.items() if key in {"field","code"} and value is not None} for item in errors)))
    def _prepared(self,lease:WorkLease,outcome:dict[str,object],config:dict[str,object])->None:
        artifacts=outcome.get("artifacts"); format_value=config["export"]["format"]
        expected={"json"} if format_value=="json" else ({"txt"} if format_value=="flat_txt" else {"json","txt"})
        if not isinstance(artifacts,list) or len(artifacts)!=len(expected): raise ExportRunnerError("export artifact list is invalid")
        actual:set[str]=set(); verified:dict[str,tuple[str,str]]={}
        for artifact in artifacts:
            if not isinstance(artifact,dict): raise ExportRunnerError("export artifact is invalid")
            kind,relative,digest=artifact.get("kind"),artifact.get("relativePath"),artifact.get("sha256")
            if kind not in expected or kind in actual or not isinstance(relative,str) or not isinstance(digest,str) or len(digest)!=64: raise ExportRunnerError("export artifact identity is invalid")
            expected_path=f"prepared\\export\\{lease.leaseId}.{kind}"
            if safe_relative_path(relative)!=expected_path: raise ExportRunnerError("export artifact path is invalid")
            path=self.view.overlay.resolve_prepared(relative)
            if not path.is_file() or sha256_file(path)!=digest: raise ExportRunnerError("export artifact digest is invalid")
            actual.add(kind); verified[kind]=(relative,digest)
        if actual!=expected: raise ExportRunnerError("export artifact format is invalid")
        if not lease.leaseId: raise ExportRunnerError("export artifact lease is missing")
        self.database.stage_export_artifacts(self.job_id,lease.sampleId,lease_id=lease.leaseId,artifacts=verified)
    def _collect_conversions(self,outcome:dict[str,object],totals:dict[str,int])->None:
        values=outcome.get("conversions")
        if values is None: return
        if not isinstance(values,dict) or not all(key in CONVERSION_CODES and type(amount) is int and amount>0 for key,amount in values.items()): raise ExportRunnerError("export conversion counts are invalid")
        if not values: return
        totals[CONVERTED_SAMPLES_CODE]=totals.get(CONVERTED_SAMPLES_CODE,0)+1
        for code,amount in values.items(): totals[code]=totals.get(code,0)+amount
    def _record_conversions(self,totals:dict[str,int])->None:
        # ROADMAP.md:988 wants an aggregated summary only, never a per-sample log.
        for code,total in totals.items():
            for start in range(0,total,MAX_PAGE_SIZE): self.database.increment_module_diagnostic(self.job_id,"export",code,severity="info",amount=min(MAX_PAGE_SIZE,total-start))
    def run(self)->str:
        config_hash,config=self._config(); active:list[WorkLease]=[]
        token_budget = config.get("tokenBudget")
        token_budget_writer = TokenBudgetOverlayWriter(self.database, self.view.overlay, self.view, self.job_id) if isinstance(token_budget, dict) and token_budget.get("enabled") is True else None
        try:
            while True:
                job=self.database.get_job(self.job_id)
                if job["status"] in {"cancelling","paused"}: return str(job["status"])
                if job["status"]!="exporting" or job["current_module_id"]!="export": raise ExportRunnerError("export module is not active")
                active=self.scheduler.claim_batch(self.job_id,"export",self.worker_instance_id,config_hash)
                if not active:
                    if self.database.count_module_unsettled(self.job_id,"export"): raise ExportRunnerError("export has unclaimable work")
                    return self.scheduler.finish_module(self.job_id,"export")
                rows=[self.database.get_leased_sample(self.job_id,"export",lease.sampleId,lease_id=str(lease.leaseId),worker_instance_id=self.worker_instance_id) for lease in active]
                if token_budget_writer is not None:
                    max_tokens = token_budget.get("maxTokens")
                    if type(max_tokens) is not int or not isinstance(config.get("captionFormat"), dict): raise ExportRunnerError("frozen Token Budget gate configuration is invalid")
                    for lease,row in zip(active,rows,strict=True):
                        try: token_budget_writer.record_for_export(sample_id=lease.sampleId,annotation_key=str(row["annotation_key"]),caption_format=config["captionFormat"],max_tokens=max_tokens)
                        except TokenBudgetOverlayError as exc: raise ExportRunnerError("Token Budget gate rejected Export") from exc
                if not self._hello:
                    payload=self._exchange("hello",{"schemaVersion":1,"payloadType":"export_hello_request","jobId":self.job_id,"configHash":config_hash,"datasetRoot":str(job["dataset_root"]),"overlayRoot":str(job["overlay_root"]),"format":config["export"]["format"],"captionFormat":config["captionFormat"]},config_hash)
                    if payload.get("schemaVersion")!=1 or payload.get("payloadType")!="export_hello_result" or payload.get("ready") is not True: raise ExportRunnerError("export hello result invalid")
                    self._hello=True
                payload=self._exchange("process_batch",{"schemaVersion":1,"payloadType":"export_process_request","items":[{"schemaVersion":1,"sampleId":lease.sampleId,"leaseId":lease.leaseId,"relativeImagePath":row["relative_image_path"],"annotationKey":row["annotation_key"]} for lease,row in zip(active,rows,strict=True)]},config_hash)
                outcomes=payload.get("outcomes")
                if payload.get("schemaVersion")!=1 or payload.get("payloadType")!="export_batch_result" or not isinstance(outcomes,list) or len(outcomes)!=len(active): raise ExportRunnerError("export batch result invalid")
                pending={(lease.sampleId,str(lease.leaseId)):(lease,row) for lease,row in zip(active,rows,strict=True)}
                conversions:dict[str,int]={}
                for outcome in outcomes:
                    if not isinstance(outcome,dict) or (outcome.get("sampleId"),outcome.get("leaseId")) not in pending: raise ExportRunnerError("export outcome identity invalid")
                    lease,row=pending.pop((outcome["sampleId"],outcome["leaseId"]));
                    if outcome.get("relativeImagePath")!=row["relative_image_path"]: raise ExportRunnerError("export path identity invalid")
                    if outcome.get("status")=="issue": self._issue(lease,row,outcome,config)
                    elif outcome.get("status")=="prepared": self._collect_conversions(outcome,conversions); self._prepared(lease,outcome,config); self.scheduler.complete(lease)
                    else: raise ExportRunnerError("export outcome status invalid")
                    active.remove(lease)
                self._record_conversions(conversions)
        except (ExportRunnerError,SchedulerError,ProtocolError):
            for lease in active: self.scheduler.release_unstarted(lease)
            self.database.set_module_summary(self.job_id,"export",status="failed",finished=True); self.database.set_job_status(self.job_id,"failed",current_module_id="export"); raise
