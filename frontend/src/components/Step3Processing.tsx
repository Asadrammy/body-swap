import { useEffect } from 'react';
import { useJobStatus } from '@/hooks/useJobStatus';
import { Loader2 } from 'lucide-react';

interface Step3ProcessingProps {
  jobId: string | null;
  onComplete: () => void;
  onError: () => void;
}

export function Step3Processing({ jobId, onComplete, onError }: Step3ProcessingProps) {
  const { data: job, error: queryError } = useJobStatus(jobId, !!jobId);

  useEffect(() => {
    if (job?.status === 'completed') {
      onComplete();
    } else if (job?.status === 'failed') {
      // Log error details for debugging
      if (job.error) {
        console.error('Job failed with error:', job.error);
      }
      onError();
    }
  }, [job?.status, job?.error, onComplete, onError]);

  // Handle query errors (e.g., network errors, 404, etc.)
  useEffect(() => {
    if (queryError) {
      console.error('Error fetching job status:', queryError);
      // Only trigger onError if we can't fetch status (e.g., job doesn't exist)
      // Don't trigger on network timeouts during processing
      if (queryError && !job) {
        onError();
      }
    }
  }, [queryError, job, onError]);

  const progress = job?.progress || 0;
  const currentStage = job?.current_stage || 'Initializing...';
  const estimatedMinutes = Math.max(1, Math.ceil((1 - progress) * 5));

  return (
    <div className="card space-y-6">
      <div>
        <h2 className="text-3xl font-semibold">Step 3: Processing Your Image</h2>
        <p className="text-slate-600 dark:text-slate-300 mt-2">
          Please wait while we create your masterpiece...
        </p>
      </div>

      <div className="processing-container rounded-3xl border border-slate-200 dark:border-white/10 bg-brandSurfaceLight/80 dark:bg-brandSurface/70 p-8 shadow-inner space-y-6">
        <div className="flex justify-center">
          <Loader2 className="h-16 w-16 text-brandAccent animate-spin" />
        </div>
        <div className="text-center text-xl font-semibold text-brandAccent dark:text-white">
          {currentStage}
        </div>
        <div className="progress-bar w-full h-4 bg-white/40 dark:bg-white/10 rounded-full overflow-hidden">
          <div
            className="progress-fill h-full bg-gradient-to-r from-brandAccent to-brandAccentAlt rounded-full transition-[width] duration-300"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
        <div className="text-center text-slate-600 dark:text-slate-300 text-sm">
          {Math.round(progress * 100)}% complete
        </div>
      </div>

      <div className="processing-info rounded-2xl bg-brandSurfaceLight dark:bg-white/5 border border-white/40 dark:border-white/5 p-6 flex flex-col gap-2 text-slate-700 dark:text-slate-200">
        <p>
          ⏱️ Estimated time: <span className="font-semibold">{estimatedMinutes}-{estimatedMinutes + 2} minutes</span>
        </p>
        <p>📧 We'll notify you when it's ready!</p>
      </div>
    </div>
  );
}


