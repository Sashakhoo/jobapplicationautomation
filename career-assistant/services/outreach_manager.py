from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session

from config import settings
from models.user import UserProfile
from models.job import JobOpportunity, JobApplication, CompanyProfile
from services.intelligence_engine import IntelligenceEngine

class OutreachManager:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.intelligence_engine = IntelligenceEngine()
    
    def generate_personalized_outreach(
        self,
        user_profile: UserProfile,
        job: JobOpportunity,
        application: Optional[JobApplication] = None
    ) -> Dict:
        """Generate personalized outreach messages"""
        outreach = {
            "cover_letter": "",
            "linkedin_message": "",
            "follow_up_email": "",
            "connection_request": ""
        }
        
        # Generate AI-powered cover letter
        if settings.OPENAI_API_KEY:
            outreach["cover_letter"] = self._generate_cover_letter(
                user_profile, job, application
            )
            
            outreach["linkedin_message"] = self._generate_linkedin_message(
                user_profile, job
            )
        
        return outreach
    
    def _generate_cover_letter(
        self,
        user_profile: UserProfile,
        job: JobOpportunity,
        application: Optional[JobApplication]
    ) -> str:
        """Generate personalized cover letter using AI"""
        prompt = f"""Write a compelling cover letter for this job application:

        Job Title: {job.title}
        Company: {job.company}
        
        Candidate Profile:
        - Name: {user_profile.user.full_name}
        - Experience: {user_profile.total_experience_years} years
        - Skills: {', '.join(user_profile.technical_skills[:10])}
        
        Job Requirements (from description):
        {job.description[:500]}
        
        The cover letter should:
        1. Be personalized to the company and role
        2. Highlight 2-3 key qualifications that match the job requirements
        3. Show enthusiasm for the company's mission/work
        4. Be concise (3-4 paragraphs)
        5. Include specific examples from the candidate's experience
        
        Tone: Professional, confident, and enthusiastic"""
        
        # In production, use LangChain or OpenAI directly
        # For now, return template
        return f"""
        Dear Hiring Manager,
        
        I am writing to express my enthusiastic interest in the {job.title} position at {job.company}. 
        With {user_profile.total_experience_years} years of experience in this field and a proven track record 
        of success in similar roles, I am confident in my ability to contribute significantly to your team.
        
        My expertise in {', '.join(user_profile.technical_skills[:3])} aligns perfectly with the requirements 
        outlined in your job description. In my previous role, I successfully [brief accomplishment].
        
        I have long admired {job.company}'s work in [specific area] and am particularly drawn to your 
        commitment to [company value from research].
        
        I am excited about the opportunity to bring my unique skills and perspective to your team and 
        contribute to [specific company goal or project].
        
        Thank you for considering my application. I look forward to discussing how I can contribute to 
        {job.company}'s continued success.
        
        Sincerely,
        {user_profile.user.full_name}
        """
    
    def _generate_linkedin_message(
        self,
        user_profile: UserProfile,
        job: JobOpportunity
    ) -> str:
        """Generate personalized LinkedIn connection/message"""
        return f"""
        Hi [Hiring Manager Name],
        
        I recently came across the {job.title} opening at {job.company} and was impressed by [specific aspect 
        of company or role].
        
        With my background in [user's field] and experience with [key skill], I believe I could make 
        valuable contributions to your team.
        
        I'd welcome the opportunity to learn more about the role and how my skills align with your needs.
        
        Best regards,
        {user_profile.user.full_name}
        """
    
    def schedule_follow_up(
        self,
        application: JobApplication,
        days: int = 7
    ) -> Dict:
        """Schedule and generate follow-up message"""
        follow_up_date = datetime.utcnow() + timedelta(days=days)
        
        message = f"""
        Subject: Following up on my application for {application.job.title}
        
        Dear [Hiring Manager Name],
        
        I hope this message finds you well. I wanted to follow up regarding my application for the 
        {application.job.title} position, which I submitted on {application.applied_date.strftime('%B %d, %Y')}.
        
        I remain very interested in this opportunity and am excited about the possibility of joining 
        {application.job.company}. The role's focus on [specific aspect] particularly resonates with my 
        experience in [relevant experience].
        
        Please let me know if you need any additional information from my side. I look forward to 
        hearing from you.
        
        Best regards,
        {application.user.full_name}
        """
        
        return {
            "scheduled_date": follow_up_date,
            "message": message,
            "action": "send_email"  # or "linkedin_message"
        }
    
    def prepare_interview_strategy(
        self,
        application: JobApplication
    ) -> Dict:
        """Prepare interview strategy based on job and company research"""
        strategy = {
            "company_research": self._research_company(application.job.company_id),
            "common_interview_questions": self._predict_interview_questions(application.job),
            "success_stories": self._identify_success_stories(application.user_profile),
            "questions_to_ask": self._generate_questions_to_ask(application.job)
        }
        
        return strategy
    
    def _research_company(self, company_id: int) -> Dict:
        """Research company for interview preparation"""
        company = self.db.query(CompanyProfile).filter_by(id=company_id).first()
        if not company:
            return {}
        
        return {
            "recent_news": f"Research recent news about {company.name}",
            "company_culture": company.core_values,
            "key_projects": "Identify recent projects or initiatives",
            "interview_process": company.interview_process
        }
    
    def _predict_interview_