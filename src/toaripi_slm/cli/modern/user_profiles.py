"""
User Profile System - Session-Only Experience Management

Provides user experience level management for adaptive CLI interface
without persistent storage. Profiles exist only during the session.
"""

from typing import List, Literal
from dataclasses import dataclass, field


@dataclass
class UserProfile:
    """Session-only user profile for personalized CLI experience."""
    
    # Identity
    display_name: str = "there"
    user_type: Literal["teacher", "student", "developer", "community_member"] = "teacher"
    
    # Experience level
    experience_level: Literal["beginner", "intermediate", "advanced", "expert"] = "beginner"
    technical_comfort: Literal["low", "medium", "high"] = "low"
    
    # Preferences
    preferred_workflow: Literal["guided", "interactive", "direct"] = "guided"
    interface_style: Literal["modern", "classic", "minimal"] = "modern"
    show_tips: bool = True
    show_progress_details: bool = True
    
    # Educational context
    primary_use_case: str = "educational_content"
    target_age_group: Literal["early_primary", "primary", "secondary", "adult"] = "primary"
    cultural_context: str = "toaripi"
    language_preference: str = "en"
    
    # Session tracking (not persisted)
    command_history: List[str] = field(default_factory=list)
    successful_workflows: List[str] = field(default_factory=list)
    
    
    def add_command_to_history(self, command: str):
        """Add command to user's session history."""
        self.command_history.append(command)
        # Keep only last 20 commands in session
        if len(self.command_history) > 20:
            self.command_history = self.command_history[-20:]
    
    def mark_workflow_completed(self, workflow: str):
        """Mark a workflow as successfully completed in this session."""
        if workflow not in self.successful_workflows:
            self.successful_workflows.append(workflow)
    
    def get_experience_description(self) -> str:
        """Get human-readable experience description."""
        descriptions = {
            "beginner": "New to AI and command-line tools",
            "intermediate": "Some experience with educational technology",
            "advanced": "Comfortable with technical tools and workflows",
            "expert": "Experienced developer or technical educator"
        }
        return descriptions.get(self.experience_level, "Unknown experience level")
    
    def get_appropriate_commands(self) -> List[str]:
        """Get commands appropriate for user's experience level."""
        
        # Base commands for all users
        base_commands = ["status", "help", "train start"]
        
        if self.experience_level == "beginner":
            return base_commands + ["data validate", "interact"]
        
        elif self.experience_level == "intermediate":
            return base_commands + [
                "data validate", "data prepare", 
                "train interactive", "model test",
                "interact", "export"
            ]
        
        elif self.experience_level == "advanced":
            return base_commands + [
                "data prepare", "data validate", "data list",
                "train start", "train interactive", "train resume",
                "model test", "model list", "model export",
                "interact", "serve start"
            ]
        
        else:  # expert
            # Show all available commands
            return [
                "data prepare", "data validate", "data list", "data clean",
                "train start", "train interactive", "train resume", "train stop",
                "model test", "model list", "model export", "model optimize",
                "interact", "serve start", "serve stop", "config", "debug"
            ]


def create_default_profile() -> UserProfile:
    """Create a default user profile for the session."""
    return UserProfile()


def create_profile_from_input(
    display_name: str = "there",
    user_type: str = "teacher", 
    experience_level: str = "beginner",
    target_age_group: str = "primary"
) -> UserProfile:
    """Create a user profile from provided parameters."""
    
    # Map experience to technical comfort
    comfort_map = {
        "beginner": "low",
        "intermediate": "medium", 
        "advanced": "medium",
        "expert": "high"
    }
    
    # Map experience to workflow preference
    workflow_map = {
        "beginner": "guided",
        "intermediate": "guided",
        "advanced": "interactive", 
        "expert": "direct"
    }
    
    return UserProfile(
        display_name=display_name,
        user_type=user_type,
        experience_level=experience_level,
        technical_comfort=comfort_map.get(experience_level, "low"),
        preferred_workflow=workflow_map.get(experience_level, "guided"),
        target_age_group=target_age_group
    )