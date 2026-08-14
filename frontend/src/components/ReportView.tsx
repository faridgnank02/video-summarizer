import type { Report } from '../api'

export function formatTs(s: number): string {
  const total = Math.floor(s)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const sec = total % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
           : `${m}:${String(sec).padStart(2, '0')}`
}

export default function ReportView({ report }: { report: Report }) {
  return (
    <article className="space-y-6">
      {report.degraded_stages.length > 0 && (
        <p className="rounded-lg bg-amber-50 px-4 py-2 text-sm text-amber-800">
          Degraded report — skipped stages: {report.degraded_stages.join(', ')}
        </p>
      )}
      <section>
        <h2 className="mb-2 text-lg font-semibold">Summary</h2>
        <p className="whitespace-pre-wrap text-slate-700">{report.summary}</p>
      </section>
      {report.chapters.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Chapters</h2>
          <ol className="space-y-1">
            {report.chapters.map((c, i) => (
              <li key={i} className="flex gap-3">
                <span className="w-16 shrink-0 font-mono text-sm text-slate-500">{formatTs(c.start_s)}</span>
                <span><strong>{c.title}</strong> — {c.synopsis}</span>
              </li>
            ))}
          </ol>
        </section>
      )}
      {report.key_quotes.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Key quotes</h2>
          <ul className="space-y-2">
            {report.key_quotes.map((q, i) => (
              <li key={i} className="border-l-2 border-slate-300 pl-3">
                <span className="mr-2 font-mono text-sm text-slate-500">{formatTs(q.timestamp_s)}</span>
                <em>"{q.text}"</em>
              </li>
            ))}
          </ul>
        </section>
      )}
      {report.action_items.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Action items</h2>
          <ul className="list-inside list-disc space-y-1">
            {report.action_items.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </section>
      )}
    </article>
  )
}
