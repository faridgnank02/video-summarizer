import { useCallback, useState } from 'react'
import { getJob, getTrace, type Job, type Trace } from './api'
import SubmitForm from './components/SubmitForm'
import JobProgress from './components/JobProgress'
import ReportView from './components/ReportView'
import TraceTable from './components/TraceTable'

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [trace, setTrace] = useState<Trace | null>(null)

  const onFinished = useCallback(async () => {
    if (jobId) {
      setJob(await getJob(jobId))
      setTrace(await getTrace(jobId))
    }
  }, [jobId])

  return (
    <main className="mx-auto max-w-3xl space-y-8 p-8">
      <h1 className="text-2xl font-semibold">Video Intelligence</h1>
      <SubmitForm onSubmitted={(id) => { setJobId(id); setJob(null); setTrace(null) }} />
      {jobId && !job && <JobProgress jobId={jobId} onFinished={onFinished} />}
      {job?.status === 'failed' && <p className="text-red-600">Failed: {job.error}</p>}
      {job?.status === 'completed' && job.report && (
        <>
          <ReportView report={job.report} />
          {trace && <TraceTable trace={trace} />}
        </>
      )}
    </main>
  )
}
