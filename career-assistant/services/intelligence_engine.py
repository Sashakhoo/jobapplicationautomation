import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from config import settings
from models.user import UserProfile
from models.job import JobOpportunity, CompanyProfile
from utils.nlp_processor import NLPProcessor

class IntelligenceEngine:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp_processor = NLPProcessor()
        
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.llm = OpenAI(temperature=0.7, max_tokens=1000)
        else:
            self.llm = None
    
    def analyze_user_profile(self, profile: UserProfile) -> Dict:
        """Deep analysis of user profile to extract strengths and opportunities"""
        analysis = {
            "strengths": [],
            "gaps": [],
            "recommendations": [],
            "market_position": {},
            "growth_areas": []
        }
        
        # Analyze skills
        if profile.technical_skills:
            skill_analysis = self._analyze_skills(profile.technical_skills)
            analysis["strengths"].extend(skill_analysis["in_demand_skills"])
            analysis["gaps"].extend(skill_analysis["missing_skills"])
        
        # Analyze experience
        if profile.experiences:
            exp_analysis = self._analyze_experience(profile.experiences)
            analysis["market_position"] = exp_analysis
        
        # Generate recommendations
        if self.llm:
            recommendations = self._generate_ai_recommendations(profile)
            analysis["recommendations"] = recommendations
        
        return analysis
    
    def _analyze_skills(self, skills: List[str]) -> Dict:
        """Analyze user skills against market demand"""
        # In a real implementation, this would compare against market data
        in_demand_skills = [
            "Python", "Machine Learning", "Cloud Computing", 
            "Data Analysis", "Project Management", "Agile Methodologies"
        ]
        
        user_skills_lower = [s.lower() for s in skills]
        in_demand_found = [
            skill for skill in in_demand_skills 
            if skill.lower() in user_skills_lower
        ]
        
        missing_skills = [
            skill for skill in in_demand_skills 
            if skill.lower() not in user_skills_lower
        ]
        
        return {
            "in_demand_skills": in_demand_found,
            "missing_skills": missing_skills,
            "skill_coverage": len(in_demand_found) / len(in_demand_skills) if in_demand_skills else 0
        }
    
    def _analyze_experience(self, experiences: List[Dict]) -> Dict:
        """Analyze work experience for patterns and progression"""
        if not experiences:
            return {}
        
        total_years = sum(exp.get("duration_years", 0) for exp in experiences)
        companies = len(experiences)
        
        # Extract roles and industries
        roles = [exp.get("title", "") for exp in experiences]
        industries = [exp.get("industry", "") for exp in experiences]
        
        # Calculate progression
        seniority_levels = self._determine_seniority(roles)
        
        return {
            "total_years": total_years,
            "companies_count": companies,
            "average_tenure_years": total_years / companies if companies > 0 else 0,
            "seniority_progression": seniority_levels,
            "industry_exposure": list(set(industries))
        }
    
    def _determine_seniority(self, roles: List[str]) -> List[str]:
        """Determine seniority level from job titles"""
        seniority_map = {
            "junior": ["junior", "associate", "trainee", "entry"],
            "mid": ["", "analyst", "specialist", "coordinator"],
            "senior": ["senior", "lead", "principal", "manager"],
            "executive": ["director", "vp", "c-level", "chief", "head"]
        }
        
        levels = []
        for role in roles:
            role_lower = role.lower()
            level = "mid"  # default
            for seniority, keywords in seniority_map.items():
                if any(keyword in role_lower for keyword in keywords if keyword):
                    level = seniority
                    break
            levels.append(level)
        
        return levels
    
    def _generate_ai_recommendations(self, profile: UserProfile) -> List[str]:
        """Generate AI-powered career recommendations"""
        prompt = PromptTemplate(
            input_variables=["profile_summary", "skills", "experience"],
            template="""As a career advisor, analyze this professional profile and provide 3-5 specific, actionable recommendations:
            
            Profile Summary: {profile_summary}
            Skills: {skills}
            Experience: {experience}
            
            Recommendations should focus on:
            1. Skill development opportunities
            2. Career progression paths
            3. Industries to target
            4. Networking strategies
            5. Portfolio improvements
            
            Provide concise, practical advice."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = chain.run({
            "profile_summary": profile.summary or "No summary provided",
            "skills": ", ".join(profile.technical_skills or []),
            "experience": f"{profile.total_experience_years} years in {len(profile.experiences or [])} roles"
        })
        
        return [rec.strip() for rec in response.split("\n") if rec.strip()]
    
    def calculate_job_match_score(
        self, 
        user_profile: UserProfile, 
        job: JobOpportunity
    ) -> Dict[str, float]:
        """Calculate comprehensive match score between user and job"""
        scores = {
            "skill_match": 0.0,
            "experience_match": 0.0,
            "culture_fit": 0.0,
            "salary_match": 0.0,
            "location_match": 0.0,
            "overall_score": 0.0
        }
        
        # Skill matching (semantic + keyword)
        if job.skill_requirements and user_profile.technical_skills:
            scores["skill_match"] = self._calculate_skill_match(
                user_profile.technical_skills,
                job.skill_requirements
            )
        
        # Experience matching
        if job.required_experience_years:
            scores["experience_match"] = self._calculate_experience_match(
                user_profile.total_experience_years,
                job.required_experience_years
            )
        
        # Calculate overall weighted score
        weights = {
            "skill_match": 0.4,
            "experience_match": 0.3,
            "culture_fit": 0.15,
            "salary_match": 0.1,
            "location_match": 0.05
        }
        
        scores["overall_score"] = sum(
            scores[key] * weights[key] 
            for key in weights.keys()
        )
        
        return scores
    
    def _calculate_skill_match(
        self, 
        user_skills: List[str], 
        job_skills: List[str]
    ) -> float:
        """Calculate skill match using semantic similarity"""
        if not job_skills:
            return 1.0
        
        # Convert to embeddings
        user_embeddings = self.embedding_model.encode(user_skills)
        job_embeddings = self.embedding_model.encode(job_skills)
        
        # Calculate cosine similarity matrix
        similarities = np.dot(user_embeddings, job_embeddings.T)
        
        # For each job skill, find best matching user skill
        best_matches = np.max(similarities, axis=0)
        
        # Average similarity across required skills
        match_score = np.mean(best_matches)
        
        return float(match_score)
    
    def _calculate_experience_match(
        self, 
        user_experience: float, 
        required_experience: float
    ) -> float:
        """Calculate experience match score"""
        if required_experience == 0:
            return 1.0
        
        # Score based on how close user experience is to requirement
        ratio = user_experience / required_experience
        
        if ratio >= 1.0:  # Overqualified
            return min(1.0, 1.5 / ratio)
        else:  # Underqualified
            return ratio  # Linear penalty for being underqualified
    
    def generate_application_strategy(
        self, 
        user_profile: UserProfile, 
        job: JobOpportunity,
        match_scores: Dict
    ) -> Dict:
        """Generate personalized application strategy"""
        strategy = {
            "strengths_to_highlight": [],
            "gaps_to_address": [],
            "key_talking_points": [],
            "resume_customization_notes": "",
            "cover_letter_focus": ""
        }
        
        # Identify key strengths based on match
        if match_scores["skill_match"] > 0.8:
            strategy["strengths_to_highlight"].append("Technical skills alignment")
        
        if match_scores["experience_match"] > 0.9:
            strategy["strengths_to_highlight"].append("Relevant experience")
        
        # Generate talking points
        if self.llm:
            strategy["key_talking_points"] = self._generate_talking_points(
                user_profile, job, match_scores
            )
        
        return strategy
    
    def _generate_talking_points(
        self, 
        user_profile: UserProfile, 
        job: JobOpportunity,
        match_scores: Dict
    ) -> List[str]:
        """Generate AI-powered talking points for interviews"""
        prompt = PromptTemplate(
            input_variables=["profile", "job_title", "company", "match_score"],
            template="""Based on this job match, generate 5 key talking points the candidate should emphasize:
            
            Job: {job_title} at {company}
            Match Score: {match_score:.0%}
            Candidate Profile: {profile}
            
            Talking points should:
            1. Highlight relevant experience
            2. Address potential gaps
            3. Show enthusiasm for the company
            4. Demonstrate cultural fit
            5. Showcase unique value proposition
            
            Return each point on a new line."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = chain.run({
            "job_title": job.title,
            "company": job.company,
            "match_score": match_scores["overall_score"],
            "profile": user_profile.summary[:500] if user_profile.summary else "No summary"
        })
        
        return [point.strip() for point in response.split("\n") if point.strip()]