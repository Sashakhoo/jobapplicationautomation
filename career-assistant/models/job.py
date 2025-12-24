from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import json

Base = declarative_base()

class JobOpportunity(Base):
    __tablename__ = "job_opportunities"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Basic Info
    title = Column(String, index=True)
    company = Column(String, index=True)
    company_linkedin_url = Column(String)
    company_career_page = Column(String)
    
    # Location
    location = Column(String)
    location_type = Column(String)  # onsite, remote, hybrid
    country = Column(String)
    state = Column(String)
    city = Column(String)
    
    # Job Details
    description = Column(Text)
    requirements = Column(Text)
    responsibilities = Column(Text)
    benefits = Column(Text)
    
    # Metadata
    seniority_level = Column(String)  # entry, mid, senior, executive
    employment_type = Column(String)  # full_time, part_time, contract, internship
    industry = Column(String)
    
    # Compensation
    salary_range_min = Column(Integer)
    salary_range_max = Column(Integer)
    salary_currency = Column(String, default="USD")
    
    # Application Info
    application_url = Column(String)
    application_method = Column(String)  # easy_apply, company_portal, email
    posted_date = Column(DateTime)
    closing_date = Column(DateTime)
    
    # Source
    source = Column(String)  # linkedin, indeed, greenhouse, lever, company_website
    source_id = Column(String, unique=True)
    original_url = Column(String)
    
    # Processing
    is_processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)
    
    # AI Analysis
    skill_requirements = Column(JSON, default=[])
    required_experience_years = Column(Float)
    education_requirements = Column(JSON, default=[])
    company_culture_keywords = Column(JSON, default=[])
    job_embedding = Column(JSON)  # Vector embedding
    
    # Quality Score
    quality_score = Column(Float, default=0.0)
    completeness_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class JobApplication(Base):
    __tablename__ = "job_applications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    job_id = Column(Integer, index=True, nullable=False)
    
    # Application Details
    status = Column(String, default="draft")  # draft, submitted, viewed, rejected, interview, offer
    applied_date = Column(DateTime)
    submitted_resume_id = Column(Integer)
    submitted_cover_letter_id = Column(Integer)
    
    # Customization
    customized_resume_text = Column(Text)
    cover_letter_text = Column(Text)
    
    # Match Analysis
    match_score = Column(Float)
    skill_match_percentage = Column(Float)
    experience_match_percentage = Column(Float)
    culture_fit_score = Column(Float)
    
    # AI Notes
    application_strategy = Column(Text)
    key_talking_points = Column(JSON, default=[])
    
    # Follow-up
    follow_up_scheduled = Column(Boolean, default=False)
    follow_up_date = Column(DateTime)
    follow_up_sent = Column(Boolean, default=False)
    
    # Tracking
    response_received = Column(Boolean, default=False)
    response_date = Column(DateTime)
    interview_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class CompanyProfile(Base):
    __tablename__ = "company_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    website = Column(String)
    linkedin_url = Column(String)
    
    # Company Info
    description = Column(Text)
    industry = Column(String)
    employee_count = Column(String)
    founded_year = Column(Integer)
    
    # Culture & Values
    mission_statement = Column(Text)
    core_values = Column(JSON, default=[])
    tech_stack = Column(JSON, default=[])
    
    # Hiring Info
    hiring_process = Column(Text)
    average_response_time_days = Column(Integer)
    interview_process = Column(Text)
    
    # AI Analysis
    culture_keywords = Column(JSON, default=[])
    growth_indicators = Column(JSON, default=[])
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())