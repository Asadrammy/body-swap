import { useState, useEffect } from 'react';
import { useJobStatus } from '@/hooks/useJobStatus';
import { swapApi } from '@/lib/api';
import { QualityMetrics } from './QualityMetrics';
import { BodySummary } from './BodySummary';
import { FitReport } from './FitReport';
import { Download, Share2, Sparkles } from 'lucide-react';

interface Step4ResultsProps {
  jobId: string | null;
  onCreateNew: () => void;
}

export function Step4Results({ jobId, onCreateNew }: Step4ResultsProps) {
  const { data: job } = useJobStatus(jobId, !!jobId);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);

  useEffect(() => {
    if (job?.status === 'completed' && jobId) {
      swapApi
        .getResult(jobId)
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          setResultImageUrl(url);
        })
        .catch((error) => {
          console.error('Error loading result:', error);
        });
    }

    return () => {
      if (resultImageUrl) {
        URL.revokeObjectURL(resultImageUrl);
      }
    };
  }, [job?.status, jobId]);

  const downloadResult = () => {
    if (!resultImageUrl || !jobId) return;
    const link = document.createElement('a');
    link.href = resultImageUrl;
    link.download = `swap_result_${jobId}.png`;
    link.click();
  };

  const downloadBundle = async () => {
    if (!jobId) return;
    try {
      const blob = await swapApi.downloadBundle(jobId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `swap_bundle_${jobId}.zip`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Bundle download failed:', error);
      alert('Unable to download bundle right now.');
    }
  };

  const shareResult = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Check out my photo swap!',
          text: 'I transformed my photo using Photo Swap Studio',
          url: window.location.href,
        });
      } catch (error) {
        // User cancelled or error
      }
    } else {
      // Fallback: copy to clipboard
      try {
        await navigator.clipboard.writeText(window.location.href);
        alert('Link copied to clipboard!');
      } catch (error) {
        console.error('Failed to copy link:', error);
      }
    }
  };

  return (
    <div className="card space-y-6">
      <h2 className="text-3xl font-semibold">Your Result is Ready! ✨</h2>

      <div className="result-container flex flex-col lg:flex-row gap-8">
        {/* Result Image */}
        <div className="flex-1 bg-brandSurfaceLight/90 dark:bg-brandSurface rounded-3xl p-6 shadow-lg border border-white/40 dark:border-white/5">
          {resultImageUrl ? (
            <img
              src={resultImageUrl}
              alt="Result"
              className="result-image w-full rounded-2xl shadow-lg mb-6 bg-slate-200/30 dark:bg-white/5 object-cover"
            />
          ) : (
            <div className="w-full aspect-square rounded-2xl bg-slate-200/30 dark:bg-white/5 flex items-center justify-center">
              <div className="text-center text-slate-500 dark:text-slate-400">
                Loading result...
              </div>
            </div>
          )}
          <div className="result-actions flex flex-wrap gap-4">
            <button onClick={downloadResult} className="btn-primary inline-flex items-center gap-2">
              <Download className="w-5 h-5" />
              Download
            </button>
            <button onClick={downloadBundle} className="btn-secondary inline-flex items-center gap-2">
              <Download className="w-5 h-5" />
              Bundle (image + metadata)
            </button>
            <button onClick={shareResult} className="btn-secondary inline-flex items-center gap-2">
              <Share2 className="w-5 h-5" />
              Share
            </button>
            <button onClick={onCreateNew} className="btn-secondary inline-flex items-center gap-2">
              <Sparkles className="w-5 h-5" />
              Create Another
            </button>
          </div>
        </div>

        {/* Metrics and Info */}
        <div className="flex-1 space-y-6">
          {job?.quality_metrics && (
            <QualityMetrics metrics={job.quality_metrics} />
          )}
          {job?.body_summary && <BodySummary summary={job.body_summary} />}
          {job?.fit_report && <FitReport report={job.fit_report} />}
        </div>
      </div>
    </div>
  );
}

