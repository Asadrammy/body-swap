import { useQuery } from '@tanstack/react-query';
import { swapApi } from '@/lib/api';
import type { JobStatus } from '@/types/api';

export function useJobStatus(jobId: string | null, enabled: boolean = true) {
  return useQuery<JobStatus>({
    queryKey: ['job', jobId],
    queryFn: () => swapApi.getStatus(jobId!),
    enabled: enabled && jobId !== null,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false; // Stop polling when done
      }
      return 3000; // Poll every 3 seconds
    },
  });
}

