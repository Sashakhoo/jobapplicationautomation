from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from models.user import UserProfile
from models.job import JobOpportunity, CompanyProfile
from services.intelligence_engine import IntelligenceEngine

class JobMatcher:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.intelligence_engine = IntelligenceEngine()
    
    def find_best_matches(
        self, 
        user_profile: UserProfile,
        limit: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Find the best job matches for a user profile"""
        # Get fresh job opportunities
        jobs = self._get_relevant_jobs(filters)
        
        # Calculate matches
        matches = []
        for job in jobs:
            match_scores = self.intelligence_engine.calculate_job_match_score(
                user_profile, job
            )
            
            if match_scores["overall_score"] >= settings.MIN_MATCH_SCORE:
                matches.append({
                    "job": job,
                    "scores": match_scores,
                    "strategy": self.intelligence_engine.generate_application_strategy(
                        user_profile, job, match_scores
                    )
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x["scores"]["overall_score"], reverse=True)
        
        return matches[:limit]
    
    def _get_relevant_jobs(self, filters: Optional[Dict] = None) -> List[JobOpportunity]:
        """Get relevant job opportunities based on filters"""
        query = self.db.query(JobOpportunity).filter(
            JobOpportunity.is_processed == True,
            JobOpportunity.posted_date >= datetime.utcnow() - timedelta(days=settings.JOB_FRESHNESS_DAYS)
        )
        
        if filters:
            if filters.get("location"):
                query = query.filter(JobOpportunity.location.contains(filters["location"]))
            
            if filters.get("job_type"):
                query = query.filter(JobOpportunity.employment_type == filters["job_type"])
            
            if filters.get("industry"):
                query = query.filter(JobOpportunity.industry == filters["industry"])
            
            if filters.get("min_salary"):
                query = query.filter(JobOpportunity.salary_range_min >= filters["min_salary"])
        
        return query.order_by(JobOpportunity.posted_date.desc()).all()
    
    def find_hidden_opportunities(
        self, 
        user_profile: UserProfile
    ) -> List[Dict]:
        """Find hidden opportunities that might not be advertised"""
        opportunities = []
        
        # 1. Companies that match user's profile
        matching_companies = self._find_matching_companies(user_profile)
        
        for company in matching_companies[:5]:
            opportunities.append({
                "type": "direct_outreach",
                "company": company,
                "reason": "Company culture and tech stack align with your profile",
                "action": "Reach out to hiring manager directly"
            })
        
        # 2. Roles based on career trajectory
        next_step_roles = self._suggest_next_step_roles(user_profile)
        
        for role in next_step_roles[:3]:
            opportunities.append({
                "type": "career_progression",
                "role": role,
                "reason": "Natural next step in your career progression",
                "action": "Search for similar roles or upskill"
            })
        
        return opportunities
    
    def _find_matching_companies(self, user_profile: UserProfile) -> List[CompanyProfile]:
        """Find companies that match user's profile and preferences"""
        # This would use semantic search on company descriptions and values
        # For now, return some example matches
        return self.db.query(CompanyProfile).limit(10).all()
    
    def _suggest_next_step_roles(self, user_profile: UserProfile) -> List[str]:
        """Suggest roles for career progression"""
        current_experience = user_profile.total_experience_years
        
        if current_experience < 3:
            return ["Senior Associate", "Team Lead", "Specialist"]
        elif current_experience < 7:
            return ["Manager", "Senior Engineer", "Product Owner"]
        else:
            return ["Director", "Principal Engineer", "Head of Department"]
    
    def analyze_market_trends(self, user_profile: UserProfile) -> Dict:
        """Analyze market trends relevant to the user's profile"""
        trends = {
            "in_demand_skills": self._get_in_demand_skills(),
            "salary_benchmarks": self._get_salary_benchmarks(user_profile),
            "hiring_trends": self._get_hiring_trends(),
            "emerging_roles": self._get_emerging_roles()
        }
        
        return trends
    
    def _get_in_demand_skills(self) -> List[str]:
        """Get currently in-demand skills"""
        # In production, this would come from job market analysis
        return [
            "AI/ML Engineering",
            "Cloud Architecture",
            "Data Engineering",
            "Cybersecurity",
            "DevOps",
            "Product Management",
            "UX/UI Design"
        ]
    
    def _get_salary_benchmarks(self, user_profile: UserProfile) -> Dict:
        """Get salary benchmarks for user's profile"""
        # Simplified version - real implementation would use market data
        experience = user_profile.total_experience_years
        
        if experience < 3:
            return {"min": 70000, "median": 85000, "max": 110000}
        elif experience < 7:
            return {"min": 100000, "median": 130000, "max": 170000}
        else:
            return {"min": 140000, "median": 180000, "max": 250000}
    
    def _get_hiring_trends(self) -> Dict:
        """Get current hiring trends"""
        return {
            "remote_work": "Stable with slight increase",
            "tech_hiring": "Moderate growth",
            "contract_roles": "Increasing",
            "time_to_hire": "Average 30-45 days"
        }
    
    def _get_emerging_roles(self) -> List[str]:
        """Get emerging job roles"""
        return [
            "AI Ethics Officer",
            "Remote Work Coordinator",
            "Sustainability Analyst",
            "Data Privacy Specialist",
            "Digital Transformation Lead"
        ]