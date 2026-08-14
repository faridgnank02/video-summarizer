import type { Trace } from '../api'

export default function TraceTable({ trace }: { trace: Trace }) {
  return (
    <section>
      <h2 className="mb-2 text-lg font-semibold">
        Trace <span className="text-sm font-normal text-slate-500">
          (total ${trace.total_cost_usd.toFixed(4)})
        </span>
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b text-slate-500">
              <th className="py-1 pr-4">Stage</th><th className="pr-4">Model</th>
              <th className="pr-4">Tokens in/out</th><th className="pr-4">Cost</th>
              <th className="pr-4">Latency</th><th>Fallback from</th>
            </tr>
          </thead>
          <tbody>
            {trace.spans.map((s, i) => (
              <tr key={i} className="border-b border-slate-100">
                <td className="py-1 pr-4">{s.stage}</td>
                <td className="pr-4 font-mono text-xs">{s.model_used}</td>
                <td className="pr-4">{s.tokens_in}/{s.tokens_out}</td>
                <td className="pr-4">${s.cost_usd.toFixed(4)}</td>
                <td className="pr-4">{s.latency_ms} ms</td>
                <td className="font-mono text-xs">{s.fallback_from ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
