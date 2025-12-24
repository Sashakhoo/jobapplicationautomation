import json
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime
import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import pdfplumber
from docx import Document
from collections import Counter

from config import settings
from models.user import UserProfile, UserDocument
from models.job import JobOpportunity
from services.intelligence_engine import IntelligenceEngine
from utils.nlp_processor import NLPProcessor

class ResumeOptimizer:
    def __init__(self):
        self.intelligence_engine = IntelligenceEngine()
        self.nlp_processor = NLPProcessor()
        
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.llm = OpenAI(temperature=0.3, max_tokens=2000)
        else:
            self.llm = None
    
    def optimize_resume_for_job(
        self,
        user_profile: UserProfile,
        original_resume_text: str,
        job: JobOpportunity,
        optimization_level: str = "aggressive"  # light, moderate, aggressive
    ) -> Dict:
        """
        Optimize resume for a specific job application
        
        Returns:
            {
                "optimized_resume": str,
                "changes_made": List[str],
                "optimization_summary": str,
                "keywords_added": List[str],
                "keywords_missing": List[str],
                "ats_score_before": float,
                "ats_score_after": float
            }
        """
        
        # Parse original resume
        resume_sections = self._parse_resume_sections(original_resume_text)
        
        # Analyze job requirements
        job_requirements = self._extract_job_requirements(job)
        
        # Calculate ATS score before optimization
        ats_score_before = self._calculate_ats_score(resume_sections, job_requirements)
        
        # Apply optimizations based on level
        if optimization_level == "light":
            optimized_sections = self._apply_light_optimization(resume_sections, job_requirements)
        elif optimization_level == "moderate":
            optimized_sections = self._apply_moderate_optimization(resume_sections, job_requirements)
        else:  # aggressive
            optimized_sections = self._apply_aggressive_optimization(resume_sections, job_requirements, user_profile)
        
        # Generate optimized resume text
        optimized_resume = self._reconstruct_resume(optimized_sections)
        
        # Calculate ATS score after optimization
        ats_score_after = self._calculate_ats_score(optimized_sections, job_requirements)
        
        # Generate summary of changes
        changes = self._generate_changes_summary(resume_sections, optimized_sections)
        
        # Identify keywords
        keywords_added, keywords_missing = self._analyze_keywords(optimized_resume, job_requirements)
        
        return {
            "optimized_resume": optimized_resume,
            "changes_made": changes,
            "optimization_summary": self._generate_optimization_summary(changes),
            "keywords_added": keywords_added,
            "keywords_missing": keywords_missing,
            "ats_score_before": ats_score_before,
            "ats_score_after": ats_score_after,
            "ats_improvement": ats_score_after - ats_score_before
        }
    
    def _parse_resume_sections(self, resume_text: str) -> Dict:
        """Parse resume into structured sections"""
        sections = {
            "header": {},
            "summary": "",
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": []
        }
        
        lines = resume_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            section_match = self._detect_section_header(line)
            if section_match:
                current_section = section_match
                continue
            
            # Add content to current section
            if current_section:
                if current_section == "header":
                    self._parse_header_line(line, sections["header"])
                elif current_section == "experience":
                    sections["experience"].append(self._parse_experience_line(line))
                elif current_section == "skills":
                    sections["skills"].extend(self._parse_skills_line(line))
                else:
                    if current_section in sections:
                        if isinstance(sections[current_section], list):
                            sections[current_section].append(line)
                        else:
                            sections[current_section] += line + "\n"
        
        # Clean up sections
        sections["experience"] = [exp for exp in sections["experience"] if exp]
        sections["skills"] = list(set(sections["skills"]))
        
        return sections
    
    def _detect_section_header(self, line: str) -> Optional[str]:
        """Detect section headers in resume"""
        section_patterns = {
            "header": r"^(name|contact|phone|email|linkedin|portfolio|website)",
            "summary": r"^(summary|profile|objective|about)$",
            "experience": r"^(experience|work history|employment|professional experience)$",
            "education": r"^(education|academic background|degrees)$",
            "skills": r"^(skills|technical skills|competencies|expertise)$",
            "projects": r"^(projects|portfolio|key projects)$",
            "certifications": r"^(certifications|certificates|licenses)$"
        }
        
        line_lower = line.lower().strip()
        
        for section, pattern in section_patterns.items():
            if re.search(pattern, line_lower):
                return section
        
        return None
    
    def _parse_header_line(self, line: str, header_dict: Dict):
        """Parse header information"""
        if "@" in line and "." in line:
            header_dict["email"] = line
        elif re.search(r'\+?[\d\s\-\(\)]+', line) and len(line.replace(" ", "")) > 7:
            header_dict["phone"] = line
        elif "linkedin.com" in line.lower():
            header_dict["linkedin"] = line
        elif "github.com" in line.lower() or "gitlab.com" in line.lower():
            header_dict["github"] = line
        elif "://" in line and not header_dict.get("portfolio"):
            header_dict["portfolio"] = line
        elif not header_dict.get("name") and len(line.split()) >= 2:
            header_dict["name"] = line
    
    def _parse_experience_line(self, line: str) -> Optional[Dict]:
        """Parse experience line into structured format"""
        # Simple parsing - in production, use more sophisticated NLP
        if len(line.split()) < 3:
            return None
        
        return {
            "text": line,
            "contains_metrics": bool(re.search(r'\d+%|\$\d+|\d+\+', line)),
            "contains_action_verbs": self._contains_action_verb(line),
            "contains_skills": self._extract_skills_from_text(line)
        }
    
    def _parse_skills_line(self, line: str) -> List[str]:
        """Parse skills from line"""
        # Split by commas, semicolons, etc.
        skills = re.split(r'[,;|•]', line)
        return [skill.strip() for skill in skills if skill.strip()]
    
    def _contains_action_verb(self, text: str) -> bool:
        """Check if text contains strong action verbs"""
        action_verbs = [
            "achieved", "improved", "increased", "reduced", "led", 
            "managed", "developed", "created", "implemented", "optimized",
            "designed", "built", "launched", "delivered", "transformed"
        ]
        
        text_lower = text.lower()
        return any(verb in text_lower for verb in action_verbs)
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills mentioned in text"""
        common_skills = [
            "python", "java", "javascript", "sql", "aws", "azure", "docker",
            "kubernetes", "react", "angular", "vue", "node", "django", "flask",
            "machine learning", "ai", "data analysis", "cloud", "git", "jenkins"
        ]
        
        text_lower = text.lower()
        found_skills = [skill for skill in common_skills if skill in text_lower]
        return found_skills
    
    def _extract_job_requirements(self, job: JobOpportunity) -> Dict:
        """Extract key requirements from job description"""
        requirements = {
            "required_skills": job.skill_requirements or [],
            "preferred_skills": [],
            "keywords": [],
            "experience_level": job.seniority_level,
            "education_requirements": job.education_requirements or []
        }
        
        # Extract keywords from job description
        if job.description:
            # Extract nouns and technical terms
            words = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]+\b', job.description)
            requirements["keywords"] = list(set(words))
            
            # Extract skills mentioned
            all_text = job.description.lower()
            technical_terms = [
                "python", "java", "javascript", "sql", "aws", "azure", 
                "docker", "kubernetes", "react", "angular", "vue",
                "machine learning", "ai", "data analysis", "cloud"
            ]
            
            for term in technical_terms:
                if term in all_text:
                    if term not in requirements["required_skills"]:
                        requirements["preferred_skills"].append(term.title())
        
        return requirements
    
    def _calculate_ats_score(self, resume_sections: Dict, job_requirements: Dict) -> float:
        """
        Calculate Applicant Tracking System (ATS) compatibility score
        
        ATS systems look for:
        1. Keyword matching
        2. Formatting (no tables, no headers/footers)
        3. Section completeness
        4. File format compatibility
        """
        score = 0.0
        max_score = 100.0
        
        # 1. Keyword matching (40%)
        resume_text = json.dumps(resume_sections).lower()
        job_keywords = [kw.lower() for kw in job_requirements["keywords"] + 
                       job_requirements["required_skills"]]
        
        if job_keywords:
            matches = sum(1 for keyword in job_keywords if keyword in resume_text)
            keyword_score = (matches / len(job_keywords)) * 40
            score += min(keyword_score, 40)
        
        # 2. Skills matching (30%)
        resume_skills = resume_sections.get("skills", [])
        required_skills = job_requirements.get("required_skills", [])
        
        if required_skills:
            resume_skills_lower = [s.lower() for s in resume_skills]
            required_skills_lower = [s.lower() for s in required_skills]
            skill_matches = sum(1 for skill in required_skills_lower 
                              if skill in resume_skills_lower)
            skill_score = (skill_matches / len(required_skills)) * 30
            score += min(skill_score, 30)
        
        # 3. Section completeness (20%)
        required_sections = ["header", "experience", "skills"]
        section_count = sum(1 for section in required_sections 
                          if resume_sections.get(section))
        section_score = (section_count / len(required_sections)) * 20
        score += section_score
        
        # 4. Experience level match (10%)
        if resume_sections.get("experience"):
            experience_years = len(resume_sections["experience"]) * 1.5  # Estimate
            if job_requirements.get("experience_level") == "senior" and experience_years >= 5:
                score += 10
            elif job_requirements.get("experience_level") == "mid" and experience_years >= 3:
                score += 10
            elif job_requirements.get("experience_level") == "entry":
                score += 10
        
        return min(score, 100.0)
    
    def _apply_light_optimization(self, resume_sections: Dict, job_requirements: Dict) -> Dict:
        """Apply minimal changes: mainly keyword optimization"""
        optimized = resume_sections.copy()
        
        # 1. Optimize summary section
        if optimized.get("summary"):
            optimized["summary"] = self._optimize_summary(
                optimized["summary"], job_requirements
            )
        
        # 2. Add missing keywords to skills section
        missing_keywords = self._identify_missing_keywords(
            optimized.get("skills", []), job_requirements
        )
        
        if missing_keywords:
            optimized["skills"] = optimized.get("skills", []) + missing_keywords[:5]
        
        # 3. Add relevant keywords to experience bullet points
        if optimized.get("experience"):
            optimized["experience"] = self._add_keywords_to_experience(
                optimized["experience"], job_requirements
            )
        
        return optimized
    
    def _apply_moderate_optimization(self, resume_sections: Dict, job_requirements: Dict) -> Dict:
        """Apply moderate changes: reordering and rewording"""
        optimized = self._apply_light_optimization(resume_sections, job_requirements)
        
        # 1. Reorder skills by relevance
        if optimized.get("skills"):
            optimized["skills"] = self._reorder_skills_by_relevance(
                optimized["skills"], job_requirements
            )
        
        # 2. Rewrite experience bullet points with stronger action verbs
        if optimized.get("experience"):
            optimized["experience"] = self._rewrite_experience_points(
                optimized["experience"], job_requirements
            )
        
        # 3. Add quantifiable achievements
        optimized["experience"] = self._add_quantifiable_achievements(
            optimized.get("experience", [])
        )
        
        return optimized
    
    def _apply_aggressive_optimization(
        self, 
        resume_sections: Dict, 
        job_requirements: Dict,
        user_profile: UserProfile
    ) -> Dict:
        """Apply aggressive changes: full restructuring and AI optimization"""
        optimized = self._apply_moderate_optimization(resume_sections, job_requirements)
        
        # 1. Use AI to rewrite entire sections
        if self.llm:
            optimized = self._ai_optimize_resume(
                optimized, job_requirements, user_profile
            )
        
        # 2. Restructure based on job requirements
        if job_requirements.get("experience_level") == "senior":
            # For senior roles: emphasize leadership and strategy
            optimized = self._optimize_for_leadership_role(optimized)
        elif "technical" in job_requirements.get("keywords", []):
            # For technical roles: emphasize skills and projects
            optimized = self._optimize_for_technical_role(optimized)
        
        # 3. Add custom summary tailored to the job
        optimized["summary"] = self._generate_targeted_summary(
            user_profile, job_requirements
        )
        
        # 4. Ensure ATS-friendly format
        optimized = self._ensure_ats_friendly_format(optimized)
        
        return optimized
    
    def _optimize_summary(self, summary: str, job_requirements: Dict) -> str:
        """Optimize professional summary with job keywords"""
        summary_lower = summary.lower()
        
        # Add missing key terms
        for keyword in job_requirements.get("keywords", [])[:3]:
            if keyword.lower() not in summary_lower:
                summary = f"{keyword} professional with " + summary
        
        # Ensure it starts with strong adjective
        strong_starts = ["Results-driven", "Experienced", "Skilled", "Accomplished"]
        if not any(summary.startswith(start) for start in strong_starts):
            summary = f"Experienced {summary}"
        
        return summary
    
    def _identify_missing_keywords(self, current_skills: List[str], job_requirements: Dict) -> List[str]:
        """Identify keywords from job requirements missing from resume"""
        current_skills_lower = [s.lower() for s in current_skills]
        missing = []
        
        for skill in job_requirements.get("required_skills", []):
            if skill.lower() not in current_skills_lower:
                missing.append(skill)
        
        for keyword in job_requirements.get("keywords", []):
            if (keyword.lower() not in current_skills_lower and 
                len(keyword.split()) == 1 and  # Single words only
                keyword.lower() not in [m.lower() for m in missing]):
                missing.append(keyword)
        
        return missing[:10]  # Limit to top 10
    
    def _add_keywords_to_experience(self, experience: List[Dict], job_requirements: Dict) -> List[Dict]:
        """Add relevant keywords to experience bullet points"""
        keywords = job_requirements.get("keywords", [])[:5]
        
        for i, exp in enumerate(experience):
            if exp and "text" in exp:
                exp_text = exp["text"].lower()
                
                # Check if any keywords are missing
                for keyword in keywords:
                    if (keyword.lower() not in exp_text and 
                        len(keyword.split()) <= 2):  # Only add short phrases
                        
                        # Find a good place to add the keyword
                        if "using" in exp_text:
                            # Add after "using"
                            parts = exp_text.split("using")
                            if len(parts) > 1:
                                new_text = parts[0] + "using " + keyword + ", " + parts[1]
                                experience[i]["text"] = new_text.capitalize()
                        elif "with" in exp_text:
                            # Add after "with"
                            parts = exp_text.split("with")
                            if len(parts) > 1:
                                new_text = parts[0] + "with " + keyword + ", " + parts[1]
                                experience[i]["text"] = new_text.capitalize()
        
        return experience
    
    def _reorder_skills_by_relevance(self, skills: List[str], job_requirements: Dict) -> List[str]:
        """Reorder skills so most relevant ones appear first"""
        required_skills = [s.lower() for s in job_requirements.get("required_skills", [])]
        preferred_skills = [s.lower() for s in job_requirements.get("preferred_skills", [])]
        
        def skill_score(skill: str) -> int:
            skill_lower = skill.lower()
            if skill_lower in required_skills:
                return 3
            elif skill_lower in preferred_skills:
                return 2
            elif any(keyword.lower() in skill_lower for keyword in job_requirements.get("keywords", [])):
                return 1
            else:
                return 0
        
        return sorted(skills, key=skill_score, reverse=True)
    
    def _rewrite_experience_points(self, experience: List[Dict], job_requirements: Dict) -> List[Dict]:
        """Rewrite experience points with stronger language and relevant keywords"""
        rewritten = []
        
        action_verb_improvements = {
            "did": "executed",
            "made": "developed",
            "helped": "contributed to",
            "worked on": "spearheaded",
            "was responsible for": "led",
            "used": "leveraged",
            "did stuff with": "optimized"
        }
        
        for exp in experience:
            if exp and "text" in exp:
                text = exp["text"]
                
                # Improve action verbs
                for weak, strong in action_verb_improvements.items():
                    if weak in text.lower():
                        text = text.lower().replace(weak, strong)
                        text = text[0].upper() + text[1:]  # Capitalize
                
                # Add quantifiers if missing
                if not exp.get("contains_metrics"):
                    # Try to add a quantifier
                    if "improved" in text.lower():
                        text = text + " by 25%"
                    elif "increased" in text.lower():
                        text = text + " by 30%"
                    elif "reduced" in text.lower():
                        text = text + " by 40%"
                
                exp["text"] = text
                rewritten.append(exp)
        
        return rewritten
    
    def _add_quantifiable_achievements(self, experience: List[Dict]) -> List[Dict]:
        """Add quantifiable metrics to experience points"""
        for exp in experience:
            if exp and "text" in exp:
                text = exp["text"].lower()
                
                # Check if already has numbers
                if not re.search(r'\d+%|\$\d+|\d+\+', text):
                    # Add appropriate metrics based on verbs
                    if any(verb in text for verb in ["increased", "improved", "grew"]):
                        exp["text"] = exp["text"] + " by 25%"
                    elif any(verb in text for verb in ["reduced", "decreased", "lowered"]):
                        exp["text"] = exp["text"] + " by 30%"
                    elif any(verb in text for verb in ["saved", "cut costs"]):
                        exp["text"] = exp["text"] + " $50K"
                    elif any(verb in text for verb in ["managed", "led", "oversaw"]):
                        exp["text"] = exp["text"] + " team of 5"
        
        return experience
    
    def _ai_optimize_resume(
        self, 
        resume_sections: Dict, 
        job_requirements: Dict,
        user_profile: UserProfile
    ) -> Dict:
        """Use AI to optimize resume sections"""
        if not self.llm:
            return resume_sections
        
        # Optimize summary with AI
        if resume_sections.get("summary"):
            prompt = PromptTemplate(
                input_variables=["original_summary", "job_requirements", "skills"],
                template="""Rewrite this professional summary to better match the job requirements:
                
                Original Summary: {original_summary}
                
                Job Requirements: {job_requirements}
                
                Candidate Skills: {skills}
                
                Make it:
                1. More compelling and achievement-oriented
                2. Include key terms from job requirements
                3. 2-3 sentences maximum
                4. Professional tone
                
                Optimized Summary:"""
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            optimized_summary = chain.run({
                "original_summary": resume_sections["summary"],
                "job_requirements": ", ".join(job_requirements.get("required_skills", [])[:10]),
                "skills": ", ".join(resume_sections.get("skills", [])[:10])
            })
            
            resume_sections["summary"] = optimized_summary.strip()
        
        # Optimize experience bullet points with AI
        if resume_sections.get("experience") and len(resume_sections["experience"]) > 0:
            optimized_experience = []
            
            for exp in resume_sections["experience"][:3]:  # Limit to first 3 for token reasons
                if exp and "text" in exp:
                    prompt = PromptTemplate(
                        input_variables=["bullet_point", "job_keywords"],
                        template="""Rewrite this resume bullet point to be more impactful and include relevant keywords:
                        
                        Original: {bullet_point}
                        
                        Job Keywords to Include: {job_keywords}
                        
                        Make it:
                        1. Start with a strong action verb
                        2. Include quantifiable results if possible
                        3. Incorporate 1-2 relevant keywords naturally
                        4. More concise and impactful
                        
                        Optimized:"""
                    )
                    
                    chain = LLMChain(llm=self.llm, prompt=prompt)
                    
                    optimized_bullet = chain.run({
                        "bullet_point": exp["text"],
                        "job_keywords": ", ".join(job_requirements.get("keywords", [])[:5])
                    })
                    
                    exp["text"] = optimized_bullet.strip()
                    optimized_experience.append(exp)
            
            resume_sections["experience"] = optimized_experience + resume_sections["experience"][3:]
        
        return resume_sections
    
    def _optimize_for_leadership_role(self, resume_sections: Dict) -> Dict:
        """Optimize resume for leadership/management roles"""
        # Move leadership experience to top
        leadership_keywords = ["led", "managed", "directed", "oversaw", "mentored", "supervised"]
        
        if resume_sections.get("experience"):
            # Separate leadership experience
            leadership_exp = []
            other_exp = []
            
            for exp in resume_sections["experience"]:
                if exp and "text" in exp:
                    text_lower = exp["text"].lower()
                    if any(keyword in text_lower for keyword in leadership_keywords):
                        leadership_exp.append(exp)
                    else:
                        other_exp.append(exp)
            
            # Put leadership experience first
            resume_sections["experience"] = leadership_exp + other_exp
        
        # Add leadership skills
        leadership_skills = [
            "Team Leadership", "Strategic Planning", "Budget Management",
            "Stakeholder Management", "Cross-functional Collaboration"
        ]
        
        current_skills = resume_sections.get("skills", [])
        for skill in leadership_skills:
            if skill not in current_skills:
                current_skills.insert(0, skill)  # Add at beginning
        
        resume_sections["skills"] = current_skills
        
        return resume_sections
    
    def _optimize_for_technical_role(self, resume_sections: Dict) -> Dict:
        """Optimize resume for technical/individual contributor roles"""
        # Add technical projects section if not present
        if not resume_sections.get("projects") and resume_sections.get("experience"):
            # Extract technical achievements from experience
            technical_keywords = ["developed", "built", "implemented", "designed", "optimized"]
            projects = []
            
            for exp in resume_sections["experience"]:
                if exp and "text" in exp:
                    text_lower = exp["text"].lower()
                    if any(keyword in text_lower for keyword in technical_keywords):
                        # Extract project-like descriptions
                        if len(exp["text"].split()) > 5:  # Not too short
                            projects.append(exp["text"])
            
            if projects:
                resume_sections["projects"] = projects[:3]  # Limit to 3 projects
        
        # Ensure technical skills are prominent
        if resume_sections.get("skills"):
            # Group technical skills
            technical_categories = {
                "Programming": ["Python", "Java", "JavaScript", "C++", "Go"],
                "Frameworks": ["React", "Django", "Spring", "TensorFlow", "PyTorch"],
                "Tools": ["Git", "Docker", "AWS", "Kubernetes", "Jenkins"],
                "Databases": ["PostgreSQL", "MongoDB", "Redis", "MySQL"]
            }
            
            organized_skills = []
            for category, skills in technical_categories.items():
                category_skills = [s for s in skills if s in resume_sections["skills"]]
                if category_skills:
                    organized_skills.append(f"{category}: {', '.join(category_skills)}")
            
            # Add other skills
            other_skills = [s for s in resume_sections["skills"] 
                          if not any(s in skill_list for skill_list in technical_categories.values())]
            if other_skills:
                organized_skills.append(f"Other: {', '.join(other_skills[:5])}")
            
            resume_sections["skills"] = organized_skills
        
        return resume_sections
    
    def _generate_targeted_summary(self, user_profile: UserProfile, job_requirements: Dict) -> str:
        """Generate a targeted professional summary for the specific job"""
        if not self.llm:
            return user_profile.summary or ""
        
        prompt = PromptTemplate(
            input_variables=["profile", "job_title", "required_skills", "experience"],
            template="""Write a compelling professional summary for a resume targeting this specific role:
            
            Target Role: {job_title}
            
            Candidate Profile:
            - Experience: {experience} years
            - Key Skills: {required_skills}
            - Background: {profile}
            
            Requirements:
            1. 2-3 sentences maximum
            2. Include the most relevant skills for this role
            3. Mention years of experience
            4. Highlight key achievements or specializations
            5. Tailor to the specific role mentioned
            
            Professional Summary:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        summary = chain.run({
            "job_title": "Senior Role" if "senior" in job_requirements.get("experience_level", "").lower() else "Role",
            "experience": user_profile.total_experience_years,
            "required_skills": ", ".join(job_requirements.get("required_skills", [])[:5]),
            "profile": user_profile.summary[:200] if user_profile.summary else "Experienced professional"
        })
        
        return summary.strip()
    
    def _ensure_ats_friendly_format(self, resume_sections: Dict) -> Dict:
        """Ensure resume format is ATS-friendly"""
        # Remove any special characters that might confuse ATS
        for section, content in resume_sections.items():
            if isinstance(content, str):
                # Remove emojis and special symbols
                resume_sections[section] = re.sub(r'[^\w\s\-\.\,\(\)]', '', content)
            elif isinstance(content, list):
                cleaned_list = []
                for item in content:
                    if isinstance(item, str):
                        cleaned_list.append(re.sub(r'[^\w\s\-\.\,\(\)]', '', item))
                    elif isinstance(item, dict) and "text" in item:
                        item["text"] = re.sub(r'[^\w\s\-\.\,\(\)]', '', item["text"])
                        cleaned_list.append(item)
                resume_sections[section] = cleaned_list
        
        return resume_sections
    
    def _reconstruct_resume(self, sections: Dict) -> str:
        """Reconstruct resume text from sections"""
        resume_lines = []
        
        # Header
        if sections.get("header"):
            header = sections["header"]
            if header.get("name"):
                resume_lines.append(header["name"])
            if header.get("email"):
                resume_lines.append(header["email"])
            if header.get("phone"):
                resume_lines.append(header["phone"])
            if header.get("linkedin"):
                resume_lines.append(header["linkedin"])
            resume_lines.append("")  # Empty line
        
        # Summary
        if sections.get("summary"):
            resume_lines.append("PROFESSIONAL SUMMARY")
            resume_lines.append("-" * 20)
            resume_lines.append(sections["summary"])
            resume_lines.append("")
        
        # Experience
        if sections.get("experience"):
            resume_lines.append("PROFESSIONAL EXPERIENCE")
            resume_lines.append("-" * 20)
            for exp in sections["experience"][:5]:  # Limit to 5 most recent
                if isinstance(exp, dict) and "text" in exp:
                    resume_lines.append(f"• {exp['text']}")
                elif isinstance(exp, str):
                    resume_lines.append(f"• {exp}")
            resume_lines.append("")
        
        # Skills
        if sections.get("skills"):
            resume_lines.append("SKILLS")
            resume_lines.append("-" * 20)
            # Format skills nicely
            skills = sections["skills"]
            if isinstance(skills[0], str) and ":" in skills[0]:
                # Already categorized
                for skill_line in skills[:6]:  # Limit to 6 categories
                    resume_lines.append(skill_line)
            else:
                # Group skills
                chunk_size = 5
                for i in range(0, min(len(skills), 15), chunk_size):
                    chunk = skills[i:i+chunk_size]
                    resume_lines.append("• " + ", ".join(chunk))
            resume_lines.append("")
        
        # Projects
        if sections.get("projects"):
            resume_lines.append("KEY PROJECTS")
            resume_lines.append("-" * 20)
            for project in sections["projects"][:3]:
                resume_lines.append(f"• {project}")
            resume_lines.append("")
        
        # Education
        if sections.get("education"):
            resume_lines.append("EDUCATION")
            resume_lines.append("-" * 20)
            for edu in sections["education"][:3]:
                resume_lines.append(f"• {edu}")
        
        return "\n".join(resume_lines)
    
    def _generate_changes_summary(self, original: Dict, optimized: Dict) -> List[str]:
        """Generate list of changes made during optimization"""
        changes = []
        
        # Check summary changes
        if original.get("summary") != optimized.get("summary"):
            changes.append("Professional summary rewritten for better impact")
        
        # Check skills changes
        orig_skills = set(str(s).lower() for s in original.get("skills", []))
        opt_skills = set(str(s).lower() for s in optimized.get("skills", []))
        added_skills = opt_skills - orig_skills
        if added_skills:
            changes.append(f"Added {len(added_skills)} relevant skills")
        
        # Check experience improvements
        orig_exp_count = len(original.get("experience", []))
        opt_exp_count = len(optimized.get("experience", []))
        if opt_exp_count > orig_exp_count:
            changes.append("Enhanced experience descriptions")
        
        # Check for quantifiable achievements
        orig_quantified = sum(1 for exp in original.get("experience", []) 
                            if isinstance(exp, dict) and exp.get("contains_metrics"))
        opt_quantified = sum(1 for exp in optimized.get("experience", []) 
                           if isinstance(exp, dict) and exp.get("contains_metrics"))
        if opt_quantified > orig_quantified:
            changes.append("Added quantifiable achievements to experience points")
        
        return changes
    
    def _generate_optimization_summary(self, changes: List[str]) -> str:
        """Generate human-readable optimization summary"""
        if not changes:
            return "No significant changes made. Resume was already well-optimized."
        
        summary = f"Optimized resume with {len(changes)} improvements:\n"
        summary += "\n".join([f"• {change}" for change in changes])
        
        if len(changes) >= 3:
            summary += "\n\nThis optimized resume is tailored to pass ATS filters and highlight your most relevant qualifications."
        
        return summary
    
    def _analyze_keywords(self, optimized_resume: str, job_requirements: Dict) -> Tuple[List[str], List[str]]:
        """Analyze keyword coverage in optimized resume"""
        resume_lower = optimized_resume.lower()
        
        # Keywords that were successfully added
        keywords_added = []
        for keyword in job_requirements.get("keywords", [])[:10]:
            if keyword.lower() in resume_lower:
                keywords_added.append(keyword)
        
        # Missing important keywords
        keywords_missing = []
        for skill in job_requirements.get("required_skills", [])[:5]:
            if skill.lower() not in resume_lower:
                keywords_missing.append(skill)
        
        return keywords_added, keywords_missing
    
    def create_resume_variants(
        self,
        user_profile: UserProfile,
        original_resume_text: str,
        target_roles: List[str]
    ) -> Dict[str, str]:
        """Create multiple resume variants for different target roles"""
        variants = {}
        
        for role in target_roles:
            # Create job requirements for this role
            job_requirements = self._create_role_requirements(role)
            
            # Optimize resume for this role
            optimized = self.optimize_resume_for_job(
                user_profile, original_resume_text,
                job_requirements, optimization_level="moderate"
            )
            
            variants[role] = optimized["optimized_resume"]
        
        return variants
    
    def _create_role_requirements(self, role: str) -> Dict:
        """Create synthetic job requirements for a role"""
        role_requirements = {
            "Software Engineer": {
                "required_skills": ["Python", "JavaScript", "SQL", "Git", "AWS"],
                "keywords": ["backend", "API", "microservices", "testing", "agile"],
                "experience_level": "mid"
            },
            "Data Scientist": {
                "required_skills": ["Python", "Machine Learning", "SQL", "Statistics", "Pandas"],
                "keywords": ["analytics", "predictive modeling", "data visualization", "AI"],
                "experience_level": "mid"
            },
            "Product Manager": {
                "required_skills": ["Product Strategy", "User Research", "Agile", "Roadmapping"],
                "keywords": ["UX", "market analysis", "stakeholder management", "MVP"],
                "experience_level": "senior"
            },
            "DevOps Engineer": {
                "required_skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Terraform"],
                "keywords": ["infrastructure", "automation", "cloud", "monitoring"],
                "experience_level": "mid"
            },
            "UX Designer": {
                "required_skills": ["Figma", "User Research", "Wireframing", "Prototyping"],
                "keywords": ["user experience", "interface design", "usability testing"],
                "experience_level": "mid"
            },
            "Data Analyst": {
                "required_skills": ["SQL", "Excel", "Tableau", "Python", "Statistics"],
                "keywords": ["data visualization", "reporting", "business intelligence"],
                "experience_level": "mid"
            }
        }
        
        return role_requirements.get(
            role, 
            {
                "required_skills": [],
                "keywords": role.split(),
                "experience_level": "mid"
            }
        )
    
    def analyze_resume_strengths(self, resume_text: str) -> Dict:
        """Analyze strengths and weaknesses of a resume"""
        sections = self._parse_resume_sections(resume_text)
        
        strengths = []
        weaknesses = []
        
        # Check summary
        if sections.get("summary"):
            if len(sections["summary"].split()) > 20:
                strengths.append("Detailed professional summary")
            else:
                weaknesses.append("Brief professional summary - consider expanding")
        else:
            weaknesses.append("Missing professional summary section")
        
        # Check experience
        if sections.get("experience"):
            exp_count = len(sections["experience"])
            if exp_count >= 3:
                strengths.append(f"Substantial experience ({exp_count} positions)")
            else:
                weaknesses.append(f"Limited experience ({exp_count} positions)")
            
            # Check for quantifiable achievements
            quantified = sum(1 for exp in sections["experience"] 
                           if isinstance(exp, dict) and exp.get("contains_metrics"))
            if quantified >= 2:
                strengths.append(f"Quantifiable achievements included ({quantified} bullet points)")
            else:
                weaknesses.append("Limited quantifiable achievements")
        else:
            weaknesses.append("Missing experience section")
        
        # Check skills
        if sections.get("skills"):
            skill_count = len(sections["skills"])
            if skill_count >= 10:
                strengths.append(f"Comprehensive skill set ({skill_count} skills)")
            else:
                weaknesses.append(f"Limited skill listing ({skill_count} skills)")
        else:
            weaknesses.append("Missing skills section")
        
        # Calculate overall score
        total_items = 7  # header, summary, experience, skills, projects, education, certifications
        completed_items = sum(1 for section, content in sections.items() 
                            if (content and (isinstance(content, list) and len(content) > 0) or 
                                (isinstance(content, str) and content.strip()) or
                                (isinstance(content, dict) and len(content) > 0)))
        
        completeness_score = (completed_items / total_items) * 100
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "completeness_score": round(completeness_score, 1),
            "section_count": completed_items,
            "recommendations": self._generate_resume_recommendations(strengths, weaknesses)
        }
    
    def _generate_resume_recommendations(self, strengths: List[str], weaknesses: List[str]) -> List[str]:
        """Generate recommendations for improving the resume"""
        recommendations = []
        
        if "Missing professional summary section" in weaknesses:
            recommendations.append("Add a professional summary highlighting your key achievements and career objectives")
        
        if "Limited quantifiable achievements" in weaknesses:
            recommendations.append("Add metrics and numbers to your experience bullet points (e.g., 'Increased sales by 30%')")
        
        if "Limited skill listing" in weaknesses:
            recommendations.append("Expand your skills section to include both technical and soft skills")
        
        # Add general recommendations
        if len(strengths) < 3:
            recommendations.append("Consider adding more detailed project descriptions or certifications")
        
        if not any("action verbs" in strength.lower() for strength in strengths):
            recommendations.append("Use stronger action verbs like 'spearheaded', 'orchestrated', 'transformed'")
        
        return recommendations
    
    def extract_resume_metrics(self, resume_text: str) -> Dict:
        """Extract key metrics and achievements from resume"""
        sections = self._parse_resume_sections(resume_text)
        
        metrics = {
            "total_experience_years": 0,
            "position_count": 0,
            "skill_count": 0,
            "quantifiable_achievements": 0,
            "education_level": "",
            "certification_count": 0
        }
        
        # Estimate total experience
        if sections.get("experience"):
            metrics["position_count"] = len(sections["experience"])
            metrics["total_experience_years"] = metrics["position_count"] * 1.5  # Rough estimate
            
            # Count quantifiable achievements
            for exp in sections["experience"]:
                if isinstance(exp, dict) and exp.get("contains_metrics"):
                    metrics["quantifiable_achievements"] += 1
        
        # Count skills
        if sections.get("skills"):
            metrics["skill_count"] = len(sections["skills"])
        
        # Check education level
        if sections.get("education"):
            education_text = " ".join(sections["education"]).lower()
            if "phd" in education_text or "doctorate" in education_text:
                metrics["education_level"] = "PhD"
            elif "master" in education_text:
                metrics["education_level"] = "Master's"
            elif "bachelor" in education_text:
                metrics["education_level"] = "Bachelor's"
            elif "associate" in education_text:
                metrics["education_level"] = "Associate's"
        
        # Count certifications
        if sections.get("certifications"):
            metrics["certification_count"] = len(sections["certifications"])
        
        return metrics