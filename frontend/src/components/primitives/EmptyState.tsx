export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-clinical border border-dashed border-clinical-line bg-clinical-raised p-6 text-center">
      <p className="font-semibold text-clinical-ink">{title}</p>
      <p className="mt-1 text-sm text-clinical-muted">{body}</p>
    </div>
  );
}

