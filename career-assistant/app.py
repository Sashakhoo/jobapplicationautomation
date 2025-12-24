from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import uvicorn
import json
import os

from config import settings
from models.user import Base, User, UserProfile, UserDocument
from models.job import JobOpportunity, JobApplication, CompanyProfile
from services.intelligence_engine import IntelligenceEngine
from services.job_matcher import JobMatcher
from services.outreach_manager import OutreachManager
from services.resume_optimizer import ResumeOptimizer
from utils.database import get_db, engine
from utils.auth import (
    authenticate_user, 
    create_access_token, 
    get_current_user,
    get_password_hash,
    verify_password
)
from utils.validators import validate_email, validate_password
from schemas import (
    UserCreate, 
    UserLogin, 
    ProfileUpdate,
    JobSearchRequest,
    JobApplicationRequest,
    ResumeOptimizationRequest
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent Career Assistant API",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Global services
intelligence_engine = IntelligenceEngine()
resume_optimizer = ResumeOptimizer()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "documentation": "/docs" if settings.DEBUG else None
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected"
    }

# ============ AUTHENTICATION ENDPOINTS ============
@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Validate email
    if not validate_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    # Validate password
    if not validate_password(user_data.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters with letters and numbers"
        )
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
        created_at=datetime.utcnow()
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Create initial profile
    profile = UserProfile(
        user_id=user.id,
        created_at=datetime.utcnow()
    )
    
    db.add(profile)
    db.commit()
    
    # Create access token
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email}
    )
    
    return {
        "message": "User registered successfully",
        "user_id": user.id,
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login user and return access token"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "full_name": user.full_name
    }

# ============ USER PROFILE ENDPOINTS ============
@app.get("/api/users/me")
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's profile"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    return {
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "location": current_user.location,
            "profile_completion": current_user.profile_completion
        },
        "profile": {
            "headline": profile.headline,
            "summary": profile.summary,
            "experiences": profile.experiences or [],
            "education": profile.education or [],
            "technical_skills": profile.technical_skills or [],
            "soft_skills": profile.soft_skills or [],
            "career_goals": profile.career_goals,
            "target_positions": profile.target_positions or []
        },
        "preferences": current_user.job_preferences
    }

@app.put("/api/users/me/profile")
async def update_user_profile(
    profile_data: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
    
    # Update basic user info
    if profile_data.full_name:
        current_user.full_name = profile_data.full_name
    if profile_data.location:
        current_user.location = profile_data.location
    
    # Update profile
    if profile_data.headline:
        profile.headline = profile_data.headline
    if profile_data.summary:
        profile.summary = profile_data.summary
    if profile_data.experiences:
        profile.experiences = profile_data.experiences
        # Calculate total experience
        total_exp = sum(exp.get("duration_years", 0) for exp in profile_data.experiences)
        profile.total_experience_years = total_exp
    if profile_data.education:
        profile.education = profile_data.education
    if profile_data.technical_skills:
        profile.technical_skills = profile_data.technical_skills
    if profile_data.soft_skills:
        profile.soft_skills = profile_data.soft_skills
    if profile_data.career_goals:
        profile.career_goals = profile_data.career_goals
    if profile_data.target_positions:
        profile.target_positions = profile_data.target_positions
    
    # Update profile completion
    completion_score = calculate_profile_completion(current_user, profile)
    current_user.profile_completion = completion_score
    current_user.last_profile_update = datetime.utcnow()
    
    db.commit()
    db.refresh(profile)
    
    return {
        "message": "Profile updated successfully",
        "profile_completion": completion_score
    }

def calculate_profile_completion(user: User, profile: UserProfile) -> float:
    """Calculate profile completion percentage"""
    completion_items = 0
    total_items = 8  # Number of profile fields
    
    if user.full_name:
        completion_items += 1
    if user.location:
        completion_items += 1
    if profile.headline:
        completion_items += 1
    if profile.summary:
        completion_items += 1
    if profile.experiences and len(profile.experiences) > 0:
        completion_items += 1
    if profile.education and len(profile.education) > 0:
        completion_items += 1
    if profile.technical_skills and len(profile.technical_skills) > 0:
        completion_items += 1
    if profile.career_goals:
        completion_items += 1
    
    return round((completion_items / total_items) * 100, 1)

# ============ JOB SEARCH ENDPOINTS ============
@app.post("/api/jobs/search")
async def search_jobs(
    search_request: JobSearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search for jobs matching user profile"""
    # Get user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Please complete your profile first"
        )
    
    # Initialize job matcher
    job_matcher = JobMatcher(db)
    
    # Apply filters
    filters = {}
    if search_request.location:
        filters["location"] = search_request.location
    if search_request.job_type:
        filters["job_type"] = search_request.job_type
    if search_request.industry:
        filters["industry"] = search_request.industry
    if search_request.min_salary:
        filters["min_salary"] = search_request.min_salary
    
    # Find matches
    matches = job_matcher.find_best_matches(
        user_profile=profile,
        limit=search_request.limit or 20,
        filters=filters if filters else None
    )
    
    # Format response
    job_results = []
    for match in matches:
        job = match["job"]
        scores = match["scores"]
        
        job_results.append({
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "location_type": job.location_type,
            "salary_range": f"${job.salary_range_min:,} - ${job.salary_range_max:,}" if job.salary_range_min else "Not specified",
            "description": job.description[:200] + "..." if job.description else "",
            "posted_date": job.posted_date.isoformat() if job.posted_date else None,
            "application_url": job.application_url,
            "match_score": round(scores["overall_score"] * 100, 1),
            "skill_match": round(scores["skill_match"] * 100, 1),
            "experience_match": round(scores["experience_match"] * 100, 1),
            "source": job.source,
            "easy_apply": job.application_method == "easy_apply"
        })
    
    # Get market insights
    market_trends = job_matcher.analyze_market_trends(profile)
    
    return {
        "jobs": job_results,
        "count": len(job_results),
        "market_trends": market_trends
    }

@app.get("/api/jobs/{job_id}")
async def get_job_details(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific job"""
    job = db.query(JobOpportunity).filter(JobOpportunity.id == job_id).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Get user profile for match calculation
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if profile:
        scores = intelligence_engine.calculate_job_match_score(profile, job)
        strategy = intelligence_engine.generate_application_strategy(profile, job, scores)
    else:
        scores = {}
        strategy = {}
    
    # Get company info
    company = db.query(CompanyProfile).filter(CompanyProfile.name == job.company).first()
    
    return {
        "job": {
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "location_type": job.location_type,
            "description": job.description,
            "requirements": job.requirements,
            "responsibilities": job.responsibilities,
            "salary_range_min": job.salary_range_min,
            "salary_range_max": job.salary_range_max,
            "salary_currency": job.salary_currency,
            "employment_type": job.employment_type,
            "seniority_level": job.seniority_level,
            "application_url": job.application_url,
            "application_method": job.application_method,
            "posted_date": job.posted_date.isoformat() if job.posted_date else None,
            "closing_date": job.closing_date.isoformat() if job.closing_date else None,
            "source": job.source,
            "skill_requirements": job.skill_requirements or []
        },
        "company": {
            "description": company.description if company else None,
            "industry": company.industry if company else None,
            "employee_count": company.employee_count if company else None,
            "website": company.website if company else None,
            "core_values": company.core_values if company else []
        } if company else None,
        "match_analysis": {
            "overall_score": round(scores.get("overall_score", 0) * 100, 1),
            "skill_match": round(scores.get("skill_match", 0) * 100, 1),
            "experience_match": round(scores.get("experience_match", 0) * 100, 1),
            "culture_fit": round(scores.get("culture_fit", 0) * 100, 1)
        },
        "application_strategy": strategy
    }

# ============ RESUME OPTIMIZATION ENDPOINTS ============
@app.post("/api/resume/optimize")
async def optimize_resume(
    request: ResumeOptimizationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Optimize resume for a specific job"""
    # Get job details
    job = db.query(JobOpportunity).filter(JobOpportunity.id == request.job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Get user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    # Get user's resume
    resume_doc = db.query(UserDocument).filter(
        UserDocument.user_id == current_user.id,
        UserDocument.document_type == "resume",
        UserDocument.is_active == True
    ).first()
    
    if not resume_doc and not request.resume_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resume found. Please upload a resume or provide resume text."
        )
    
    # Get resume text
    resume_text = request.resume_text
    if not resume_text and resume_doc:
        # Read resume file (simplified - in production, read actual file)
        resume_text = "Resume content would be here from file"
    
    # Optimize resume
    optimization_result = resume_optimizer.optimize_resume_for_job(
        user_profile=profile,
        original_resume_text=resume_text,
        job=job,
        optimization_level=request.optimization_level or "moderate"
    )
    
    # Save optimized resume as a new document
    optimized_doc = UserDocument(
        user_id=current_user.id,
        document_type="optimized_resume",
        file_name=f"resume_optimized_{job.company}_{datetime.utcnow().date()}.txt",
        metadata={
            "job_id": job.id,
            "job_title": job.title,
            "company": job.company,
            "optimization_level": request.optimization_level,
            "ats_score_before": optimization_result["ats_score_before"],
            "ats_score_after": optimization_result["ats_score_after"]
        },
        created_at=datetime.utcnow()
    )
    
    db.add(optimized_doc)
    db.commit()
    
    return {
        "optimized_resume": optimization_result["optimized_resume"],
        "optimization_id": optimized_doc.id,
        "changes_made": optimization_result["changes_made"],
        "keywords_added": optimization_result["keywords_added"],
        "keywords_missing": optimization_result["keywords_missing"],
        "ats_score_before": optimization_result["ats_score_before"],
        "ats_score_after": optimization_result["ats_score_after"],
        "ats_improvement": optimization_result["ats_improvement"],
        "summary": optimization_result["optimization_summary"]
    }

@app.post("/api/resume/variants")
async def create_resume_variants(
    target_roles: List[str],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create multiple resume variants for different target roles"""
    # Get user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    # Get user's resume
    resume_doc = db.query(UserDocument).filter(
        UserDocument.user_id == current_user.id,
        UserDocument.document_type == "resume",
        UserDocument.is_active == True
    ).first()
    
    if not resume_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No resume found. Please upload a resume first."
        )
    
    # Create variants
    variants = resume_optimizer.create_resume_variants(
        user_profile=profile,
        original_resume_text="Resume content would be here",  # In production, read actual file
        target_roles=target_roles
    )
    
    # Save variants as documents
    variant_docs = []
    for role, content in variants.items():
        variant_doc = UserDocument(
            user_id=current_user.id,
            document_type="resume_variant",
            file_name=f"resume_{role.replace(' ', '_').lower()}.txt",
            metadata={
                "target_role": role,
                "created_at": datetime.utcnow().isoformat()
            },
            created_at=datetime.utcnow()
        )
        db.add(variant_doc)
        variant_docs.append({
            "role": role,
            "document_id": variant_doc.id,
            "content_preview": content[:200] + "..." if len(content) > 200 else content
        })
    
    db.commit()
    
    return {
        "variants": variant_docs,
        "count": len(variant_docs)
    }

# ============ APPLICATION ENDPOINTS ============
@app.post("/api/applications")
async def create_application(
    request: JobApplicationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new job application"""
    # Check if job exists
    job = db.query(JobOpportunity).filter(JobOpportunity.id == request.job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Check if already applied
    existing_application = db.query(JobApplication).filter(
        JobApplication.user_id == current_user.id,
        JobApplication.job_id == request.job_id
    ).first()
    
    if existing_application:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already applied to this job"
        )
    
    # Get user profile for match calculation
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if profile:
        match_scores = intelligence_engine.calculate_job_match_score(profile, job)
        match_score = match_scores["overall_score"]
    else:
        match_score = 0
    
    # Create application record
    application = JobApplication(
        user_id=current_user.id,
        job_id=request.job_id,
        status="applied",
        applied_date=datetime.utcnow(),
        match_score=match_score,
        skill_match_percentage=match_scores.get("skill_match", 0) if profile else 0,
        experience_match_percentage=match_scores.get("experience_match", 0) if profile else 0,
        created_at=datetime.utcnow()
    )
    
    db.add(application)
    db.commit()
    db.refresh(application)
    
    # Generate outreach materials if requested
    outreach_materials = None
    if request.generate_materials and profile:
        outreach_manager = OutreachManager(db)
        outreach_materials = outreach_manager.generate_personalized_outreach(
            user_profile=profile,
            job=job,
            application=application
        )
    
    return {
        "message": "Application created successfully",
        "application_id": application.id,
        "match_score": round(match_score * 100, 1),
        "outreach_materials": outreach_materials,
        "next_steps": [
            "Save any custom cover letters or messages",
            "Schedule a follow-up for 7 days from now",
            "Prepare for potential interview questions"
        ]
    }

@app.get("/api/applications")
async def get_user_applications(
    status_filter: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's job applications"""
    query = db.query(JobApplication).filter(JobApplication.user_id == current_user.id)
    
    if status_filter:
        query = query.filter(JobApplication.status == status_filter)
    
    applications = query.order_by(JobApplication.applied_date.desc()).offset(offset).limit(limit).all()
    
    application_list = []
    for app in applications:
        job = db.query(JobOpportunity).filter(JobOpportunity.id == app.job_id).first()
        
        application_list.append({
            "id": app.id,
            "job_title": job.title if job else "Unknown",
            "company": job.company if job else "Unknown",
            "applied_date": app.applied_date.isoformat() if app.applied_date else None,
            "status": app.status,
            "match_score": round(app.match_score * 100, 1) if app.match_score else None,
            "response_received": app.response_received,
            "interview_count": app.interview_count,
            "follow_up_scheduled": app.follow_up_scheduled
        })
    
    # Get statistics
    total_applications = db.query(JobApplication).filter(
        JobApplication.user_id == current_user.id
    ).count()
    
    interviews_scheduled = db.query(JobApplication).filter(
        JobApplication.user_id == current_user.id,
        JobApplication.status == "interview"
    ).count()
    
    offers_received = db.query(JobApplication).filter(
        JobApplication.user_id == current_user.id,
        JobApplication.status == "offer"
    ).count()
    
    return {
        "applications": application_list,
        "pagination": {
            "total": total_applications,
            "limit": limit,
            "offset": offset
        },
        "statistics": {
            "total_applications": total_applications,
            "interviews_scheduled": interviews_scheduled,
            "offers_received": offers_received,
            "interview_rate": round((interviews_scheduled / total_applications * 100), 1) if total_applications > 0 else 0,
            "offer_rate": round((offers_received / total_applications * 100), 1) if total_applications > 0 else 0
        }
    }

# ============ INTELLIGENCE & ANALYTICS ENDPOINTS ============
@app.get("/api/analytics/profile")
async def get_profile_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed analytics about user's profile and market position"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    # Analyze profile
    profile_analysis = intelligence_engine.analyze_user_profile(profile)
    
    # Get market trends
    job_matcher = JobMatcher(db)
    market_trends = job_matcher.analyze_market_trends(profile)
    
    # Get hidden opportunities
    hidden_opportunities = job_matcher.find_hidden_opportunities(profile)
    
    return {
        "profile_analysis": profile_analysis,
        "market_trends": market_trends,
        "hidden_opportunities": hidden_opportunities,
        "profile_strength": current_user.profile_completion,
        "last_updated": current_user.last_profile_update.isoformat() if current_user.last_profile_update else None
    }

@app.get("/api/analytics/skills")
async def get_skills_analysis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze user's skills against market demand"""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile or not profile.technical_skills:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile or skills not found"
        )
    
    # Get in-demand skills (simplified - in production, use real market data)
    in_demand_skills = [
        "Python", "Machine Learning", "AWS", "Docker", "Kubernetes",
        "React", "Node.js", "SQL", "Data Analysis", "Cloud Computing"
    ]
    
    user_skills = profile.technical_skills
    matched_skills = [skill for skill in user_skills if skill in in_demand_skills]
    missing_skills = [skill for skill in in_demand_skills if skill not in user_skills]
    
    # Calculate skill marketability score
    skill_coverage = len(matched_skills) / len(in_demand_skills) * 100 if in_demand_skills else 0
    
    return {
        "user_skills": user_skills,
        "in_demand_skills": in_demand_skills,
        "matched_skills": matched_skills,
        "missing_high_value_skills": missing_skills,
        "skill_coverage_percentage": round(skill_coverage, 1),
        "recommendations": [
            f"Consider learning: {', '.join(missing_skills[:3])}" if missing_skills else "Your skills are well-aligned with market demand",
            f"Highlight these in-demand skills: {', '.join(matched_skills[:5])}"
        ]
    }

# ============ ADMIN ENDPOINTS (Development Only) ============
if settings.DEBUG:
    @app.get("/api/admin/users")
    async def get_all_users(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """Get all users (admin only)"""
        # Simple admin check - in production, implement proper admin auth
        if current_user.email != "sashakhoo8@gmail.com":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        users = db.query(User).all()
        return {"users": [
            {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "profile_completion": user.profile_completion
            } for user in users
        ]}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred"},
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )