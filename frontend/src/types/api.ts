// API Types matching FastAPI schemas

export interface SwapRequest {
  customer_photos: string[];
  template_image: string;
  refinement_mask?: string;
  options?: Record<string, unknown>;
}

export interface SwapResponse {
  job_id: string;
  status: string;
  message?: string;
  created_at: string;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number; // 0-1
  current_stage?: string;
  error?: string;
  body_summary?: BodySummary;
  fit_report?: FitReport;
  quality_metrics?: QualityMetrics;
  refinement_masks?: Record<string, string>;
  created_at: string;
  updated_at: string;
}

export interface BodySummary {
  body_type?: string;
  measurements?: Record<string, number>;
  confidence?: number;
}

export interface FitReport {
  scale_map?: Record<string, number>;
  items?: Record<string, FitItem>;
  skin_synthesis_applied?: boolean;
}

export interface FitItem {
  status: string;
  scale_x?: number;
  scale_y?: number;
}

export interface QualityMetrics {
  overall_score?: number;
  face_similarity?: number;
  pose_accuracy?: number;
  clothing_fit?: number;
  seamless_blending?: number;
  sharpness?: number;
  recommended_refinements?: string[];
}

export interface TemplateMetadata {
  id: string;
  name: string;
  category: string;
  description?: string;
  preview_url: string;
  asset_path: string;
  pose_type?: string;
  clothing_style?: string;
  body_visibility?: string;
  recommended_subjects: string[];
  background?: string;
  tags: string[];
}

export interface TemplateListResponse {
  templates: TemplateMetadata[];
  total: number;
}

export interface RefineRequest {
  job_id: string;
  refinement_mask?: string;
  regions?: string[];
  strength?: number;
}

