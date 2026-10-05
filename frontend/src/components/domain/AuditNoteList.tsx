import { Card } from "../primitives/Card";

export function AuditNoteList({ notes }: { notes: string[] }) {
  return (
    <Card title="Evidence log" eyebrow="Audit notes">
      <ul className="space-y-2">
        {notes.map((note) => (
          <li key={note} className="rounded-md border border-clinical-line bg-clinical-raised px-3 py-2 text-sm text-clinical-ink">
            {note}
          </li>
        ))}
      </ul>
    </Card>
  );
}

