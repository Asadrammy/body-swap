import { useQuery } from '@tanstack/react-query';
import { templatesApi } from '@/lib/api';
import type { TemplateListResponse } from '@/types/api';

export function useTemplates(category?: string, tag?: string) {
  return useQuery<TemplateListResponse>({
    queryKey: ['templates', category, tag],
    queryFn: () => templatesApi.list(category, tag),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

