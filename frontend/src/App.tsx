import { useCallback, useState } from 'react'
import { getJob, type Job } from './api'
import SubmitForm from './components/SubmitForm'
import JobProgress from './components/JobProgress'

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)

  const onFinished = useCallback(async () => {
    if (jobId) setJob(await getJob(jobId))
  }, [jobId])

  return (
    <main className="mx-auto max-w-3xl space-y-8 p-8">
      <h1 className="text-2xl font-semibold">Video Intelligence</h1>
      <SubmitForm onSubmitted={(id) => { setJobId(id); setJob(null) }} />
      {jobId && !job && <JobProgress jobId={jobId} onFinished={onFinished} />}
      {job?.status === 'failed' && <p className="text-red-600">Failed: {job.error}</p>}
      {job?.status === 'completed' && job.report && (
        <pre className="overflow-x-auto rounded-lg bg-slate-100 p-4 text-xs">
          {JSON.stringify(job.report, null, 2)}
        </pre> /* replaced by ReportView in the next task */
      )}
    </main>
  )
}
