import type { JobSnapshot } from "../api";

type RepairTask = JobSnapshot["repairChildren"][number];

export type RepairTasksPanelProps = {
  tasks: ReadonlyArray<RepairTask>;
  pending: boolean;
  labels: {
    title: string;
    open: string;
    delete: string;
    deleteCompletedHistory: string;
    targets: (sampleCount: number, targetCount: number) => string;
    createdAt: (createdAt: string) => string;
  };
  onOpen: (jobId: string) => void;
  onDelete: (jobId: string, status: string) => void;
};

export function RepairTasksPanel({ tasks, pending, labels, onOpen, onDelete }: RepairTasksPanelProps) {
  return <section className="repair-tasks-panel" aria-labelledby="repair-tasks-heading">
    <div className="panel-heading">
      <h3 id="repair-tasks-heading">{labels.title}</h3>
    </div>
    <ul className="repair-task-list">
      {tasks.map((task) => <li key={task.jobId}>
        <div className="repair-task-summary">
          <strong><code>{task.jobId}</code></strong>
          <span className={`status ${task.status}`}>{task.status}</span>
          <small>{labels.targets(task.sampleCount, task.targetCount)}</small>
          <small>{labels.createdAt(task.createdAt)}</small>
        </div>
        <div className="repair-task-actions">
          <button className="secondary" type="button" onClick={() => onOpen(task.jobId)}>{labels.open}</button>
          <button className="danger-action" type="button" disabled={pending} aria-busy={pending} onClick={() => onDelete(task.jobId, task.status)}>{task.status === "succeeded" ? labels.deleteCompletedHistory : labels.delete}</button>
        </div>
      </li>)}
    </ul>
  </section>;
}
