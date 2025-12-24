from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import json

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    phone = Column(String)
    location = Column(String)
    timezone = Column(String, default="UTC")
    
    # Profile Strength
    profile_completion = Column(Float, default=0.0)
    last_profile_update = Column(DateTime, default=func.now())
    
    # Preferences
    job_preferences = Column(JSON, default={
        "roles": [],
        "industries": [],
        "company_size": [],
        "work_arrangement": ["hybrid", "remote"],
        "salary_range": {"min": 60000, "max": 150000},
        "seniority_level": []
    })
    
    application_preferences = Column(JSON, default={
        "daily_limit": 5,
        "min_match_score": 0.8,
        "require_cover_letter": True,
        "customize_resume": True,
        "follow_up_days": 7
    })
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    
    # Professional Summary
    headline = Column(String)
    summary = Column(Text)
    
    # Experience
    experiences = Column(JSON, default=[])
    total_experience_years = Column(Float, default=0.0)
    
    # Education
    education = Column(JSON, default=[])
    
    # Skills
    technical_skills = Column(JSON, default=[])
    soft_skills = Column(JSON, default=[])
    certifications = Column(JSON, default=[])
    
    # Career Goals
    career_goals = Column(Text)
    target_positions = Column(JSON, default=[])
    
    # Resume Data
    resume_text = Column(Text)
    resume_embedding = Column(JSON)  # Vector embedding for semantic search
    
    # Performance Metrics
    success_rate = Column(Float, default=0.0)
    average_match_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class UserDocument(Base):
    __tablename__ = "user_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    document_type = Column(String)  # resume, cover_letter_template, portfolio
    file_name = Column(String)
    file_path = Column(String)
    content_hash = Column(String)
    metadata = Column(JSON, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())