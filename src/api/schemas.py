"""API request/response schemas"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class SwapRequest(BaseModel):
    """Request schema for face/body swap"""
    customer_photos: List[str] = Field(..., description="Paths to customer reference photos (1-2 images)")
    template_image: str = Field(..., description="Path to template image")
    refinement_mask: Optional[str] = Field(None, description="Optional mask for selective refinement")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")


class SwapResponse(BaseModel):
    """Response schema for swap request"""
    job_id: str = Field(..., description="Job ID for tracking")
    status: str = Field(..., description="Job status")
    message: Optional[str] = Field(None, description="Status message")
    created_at: datetime = Field(default_factory=datetime.now)


class JobStatus(BaseModel):
    """Job status schema"""
    job_id: str
    status: str = Field(..., description="pending, processing, completed, failed")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress percentage (0-1)")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    error: Optional[str] = Field(None, description="Error message if failed")
    body_summary: Optional[Dict[str, Any]] = Field(None, description="Aggregated body measurements summary")
    fit_report: Optional[Dict[str, Any]] = Field(None, description="Clothing adaptation report")
    quality_metrics: Optional[Dict[str, Any]] = Field(None, description="Quality evaluation metrics")
    refinement_masks: Optional[Dict[str, str]] = Field(None, description="Paths to refinement masks")
    created_at: datetime
    updated_at: datetime


class RefineRequest(BaseModel):
    """Request schema for refinement"""
    job_id: str = Field(..., description="Original job ID")
    refinement_mask: Optional[str] = Field(None, description="Mask for selective refinement")
    regions: Optional[List[str]] = Field(None, description="Regions to refine (face, body, clothing, etc.)")
    strength: Optional[float] = Field(0.8, ge=0.0, le=1.0, description="Refinement strength")


class ResultResponse(BaseModel):
    """Response schema for result download"""
    job_id: str
    result_image: str = Field(..., description="Path to result image")
    intermediate_results: Optional[Dict[str, str]] = Field(None, description="Paths to intermediate results")
    quality_metrics: Optional[Dict[str, float]] = Field(None, description="Quality assessment metrics")
    refinement_masks: Optional[Dict[str, str]] = Field(None, description="Paths to refinement masks")


class TemplateMetadata(BaseModel):
    """Metadata describing a template entry."""

    id: str
    name: str
    category: str
    description: Optional[str]
    preview_url: str
    asset_path: str
    pose_type: Optional[str]
    clothing_style: Optional[str]
    body_visibility: Optional[str]
    recommended_subjects: List[str] = Field(default_factory=list)
    background: Optional[str]
    tags: List[str] = Field(default_factory=list)


class TemplateListResponse(BaseModel):
    """Response payload for template catalog."""

    templates: List[TemplateMetadata]
    total: int

