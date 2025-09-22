"""
Smart Guidance System - Intelligent Help and Command Suggestion

Provides contextual help, command suggestions, and next step recommendations
based on user profile, current project state, and educational objectives.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

from .framework import CLIContext
from .user_profiles import UserProfile


@dataclass
class SuggestedAction:
    """Represents a suggested action for the user."""
    
    title: str
    description: str
    command: str
    emoji: str
    priority: int = 1  # 1 = highest priority
    difficulty: str = "easy"  # easy, medium, hard
    estimated_time: str = "5 minutes"
    prerequisites: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class HelpContent:
    """Represents contextual help content."""
    
    title: str
    content: str
    examples: List[str]
    related_commands: List[str]
    tips: List[str]


class SmartGuidance:
    """Provides intelligent guidance and suggestions based on context."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self._action_templates = self._initialize_action_templates()
        self._help_content = self._initialize_help_content()
    
    def suggest_next_actions(self, max_suggestions: int = 3) -> List[SuggestedAction]:
        """Suggest next actions based on current context and user profile."""
        
        # Get current project state
        project_state = self._analyze_project_state()
        
        # Get user profile
        profile = self.context.user_profile
        if not profile:
            profile = UserProfile()  # Default profile
        
        # Generate suggestions based on state
        suggestions = []
        
        # No training data
        if not project_state.get("has_training_data", False):
            suggestions.extend(self._get_data_preparation_suggestions(profile))
        
        # Has data but no models
        elif not project_state.get("has_trained_models", False):
            suggestions.extend(self._get_training_suggestions(profile))
        
        # Has models - suggest usage
        else:
            suggestions.extend(self._get_usage_suggestions(profile))
        
        # Add universal suggestions
        suggestions.extend(self._get_universal_suggestions(profile))
        
        # Filter by user experience level and sort by priority
        filtered_suggestions = self._filter_by_experience(suggestions, profile)
        filtered_suggestions.sort(key=lambda x: x.priority)
        
        return filtered_suggestions[:max_suggestions]
    
    def _analyze_project_state(self) -> Dict[str, Any]:
        """Analyze current project state to inform suggestions."""
        
        state = {
            "has_training_data": False,
            "has_trained_models": False,
            "has_config_files": False,
            "data_file_count": 0,
            "model_file_count": 0,
            "last_training_session": None
        }
        
        # Check for training data
        data_dir = self.context.working_directory / "data"
        if data_dir.exists():
            data_files = list(data_dir.rglob("*.csv"))
            state["data_file_count"] = len(data_files)
            state["has_training_data"] = len(data_files) > 0
        
        # Check for trained models
        models_dir = self.context.working_directory / "models"
        if models_dir.exists():
            model_files = list(models_dir.rglob("*.bin")) + list(models_dir.rglob("*.gguf"))
            state["model_file_count"] = len(model_files)
            state["has_trained_models"] = len(model_files) > 0
        
        # Check for config files
        configs_dir = self.context.working_directory / "configs"
        if configs_dir.exists():
            config_files = list(configs_dir.rglob("*.yaml")) + list(configs_dir.rglob("*.yml"))
            state["has_config_files"] = len(config_files) > 0
        
        return state
    
    def _get_data_preparation_suggestions(self, profile: UserProfile) -> List[SuggestedAction]:
        """Get suggestions for data preparation phase."""
        
        suggestions = [
            SuggestedAction(
                title="Prepare training data",
                description="Upload and validate your English-Toaripi parallel text",
                command="data prepare",
                emoji="📚",
                priority=1,
                estimated_time="5 minutes"
            )
        ]
        
        # Add beginner-friendly explanation
        if profile.experience_level == "beginner":
            suggestions[0].description += " (guided step-by-step)"
            suggestions[0].command = "data prepare --guided"
        
        return suggestions
    
    def _get_training_suggestions(self, profile: UserProfile) -> List[SuggestedAction]:
        """Get suggestions for training phase."""
        
        suggestions = []
        
        if profile.experience_level == "beginner":
            suggestions.append(SuggestedAction(
                title="Train your first model",
                description="Create an educational content model with guided setup",
                command="train start --beginner",
                emoji="🚀",
                priority=1,
                estimated_time="15-30 minutes"
            ))
        else:
            suggestions.append(SuggestedAction(
                title="Start training",
                description="Train a model for educational content generation",
                command="train start",
                emoji="🚀", 
                priority=1,
                estimated_time="15-60 minutes"
            ))
        
        # Add validation suggestion
        suggestions.append(SuggestedAction(
            title="Validate your data first",
            description="Check data quality before training",
            command="data validate",
            emoji="✅",
            priority=2,
            estimated_time="2 minutes"
        ))
        
        return suggestions
    
    def _get_usage_suggestions(self, profile: UserProfile) -> List[SuggestedAction]:
        """Get suggestions for users with trained models."""
        
        return [
            SuggestedAction(
                title="Generate educational content",
                description="Create stories, vocabulary, and exercises in Toaripi",
                command="interact",
                emoji="📝",
                priority=1,
                estimated_time="ongoing"
            ),
            SuggestedAction(
                title="Test your model",
                description="Evaluate model performance with sample content",
                command="model test",
                emoji="🧪",
                priority=2,
                estimated_time="5 minutes"
            ),
            SuggestedAction(
                title="Export for classroom use",
                description="Create offline-ready model for deployment",
                command="model export --format gguf",
                emoji="📦",
                priority=3,
                estimated_time="10 minutes"
            )
        ]
    
    def _get_universal_suggestions(self, profile: UserProfile) -> List[SuggestedAction]:
        """Get suggestions that are always relevant."""
        
        suggestions = [
            SuggestedAction(
                title="View system status",
                description="Check current status and health",
                command="status",
                emoji="📊",
                priority=5,
                estimated_time="1 minute"
            )
        ]
        
        # Add help for beginners
        if profile.experience_level == "beginner":
            suggestions.append(SuggestedAction(
                title="Get help and tutorials",
                description="Learn how to use the system effectively",
                command="help --guide",
                emoji="❓",
                priority=4,
                estimated_time="5 minutes"
            ))
        
        return suggestions
    
    def _filter_by_experience(self, suggestions: List[SuggestedAction], profile: UserProfile) -> List[SuggestedAction]:
        """Filter suggestions based on user experience level."""
        
        # For beginners, only show easy tasks
        if profile.experience_level == "beginner":
            return [s for s in suggestions if s.difficulty in ["easy"]]
        
        # For intermediate users, show easy and medium
        elif profile.experience_level == "intermediate":
            return [s for s in suggestions if s.difficulty in ["easy", "medium"]]
        
        # For advanced/expert users, show all
        else:
            return suggestions
    
    def parse_natural_language_command(self, input_text: str) -> Optional[str]:
        """Parse natural language input to suggest commands."""
        
        input_lower = input_text.lower().strip()
        
        # Training keywords
        if any(keyword in input_lower for keyword in [
            "train", "create model", "build model", "start training", "make model"
        ]):
            return "train start"
        
        # Data keywords  
        if any(keyword in input_lower for keyword in [
            "check data", "validate data", "prepare data", "upload data", "data"
        ]):
            if "prepare" in input_lower or "upload" in input_lower:
                return "data prepare"
            else:
                return "data validate"
        
        # Generation keywords
        if any(keyword in input_lower for keyword in [
            "generate", "create story", "make content", "write story", "interact"
        ]):
            return "interact"
        
        # Testing keywords
        if any(keyword in input_lower for keyword in [
            "test", "evaluate", "check model", "how good"
        ]):
            return "model test"
        
        # Status keywords
        if any(keyword in input_lower for keyword in [
            "status", "health", "check system", "what's wrong", "problems"
        ]):
            return "status"
        
        # Help keywords
        if any(keyword in input_lower for keyword in [
            "help", "how to", "what can", "guide", "tutorial", "learn"
        ]):
            return "help"
        
        # Export keywords
        if any(keyword in input_lower for keyword in [
            "export", "deploy", "raspberry pi", "offline", "classroom"
        ]):
            return "model export"
        
        return None
    
    def get_contextual_help(self, command: str, context: str = "") -> Optional[HelpContent]:
        """Get contextual help for a specific command."""
        
        # This would be expanded with comprehensive help content
        help_map = {
            "train": HelpContent(
                title="Training Models",
                content="Train AI models to generate educational content in Toaripi language.",
                examples=[
                    "toaripi train start --beginner",
                    "toaripi train start --data custom_data.csv",
                    "toaripi train resume last_session"
                ],
                related_commands=["data validate", "model test", "status"],
                tips=[
                    "Start with 'train start --beginner' for guided setup",
                    "Validate your data first with 'data validate'",
                    "Training time depends on data size and computer speed"
                ]
            ),
            "data": HelpContent(
                title="Data Management", 
                content="Prepare and validate English-Toaripi parallel text for training.",
                examples=[
                    "toaripi data prepare bible_texts.csv",
                    "toaripi data validate --detailed",
                    "toaripi data list"
                ],
                related_commands=["train start", "status"],
                tips=[
                    "Use CSV files with 'english' and 'toaripi' columns",
                    "Cultural appropriateness is automatically checked",
                    "Larger datasets generally produce better models"
                ]
            ),
            "interact": HelpContent(
                title="Interactive Generation",
                content="Generate educational content through conversational interface.",
                examples=[
                    "toaripi interact",
                    "toaripi interact --content-type story", 
                    "toaripi interact --version v1.0.0"
                ],
                related_commands=["model test", "train start"],
                tips=[
                    "Ask for stories, vocabulary, or dialogues",
                    "Content is automatically validated for age-appropriateness",
                    "Use /help within interactive mode for commands"
                ]
            )
        }
        
        return help_map.get(command)
    
    def _initialize_action_templates(self) -> Dict[str, SuggestedAction]:
        """Initialize templates for common actions."""
        
        # This could be loaded from configuration files
        return {}
    
    def _initialize_help_content(self) -> Dict[str, HelpContent]:
        """Initialize help content database."""
        
        # This could be loaded from documentation files
        return {}


class GuidanceEngine:
    """Main guidance engine that coordinates all guidance features."""
    
    def __init__(self, context: CLIContext):
        self.context = context
        self.smart_guidance = SmartGuidance(context)
    
    def suggest_next_actions(self, max_suggestions: int = 3) -> List[SuggestedAction]:
        """Get next action suggestions."""
        return self.smart_guidance.suggest_next_actions(max_suggestions)
    
    def parse_natural_language(self, input_text: str) -> Optional[str]:
        """Parse natural language to command suggestion."""
        return self.smart_guidance.parse_natural_language_command(input_text)
    
    def get_help(self, command: str, context: str = "") -> Optional[HelpContent]:
        """Get contextual help."""
        return self.smart_guidance.get_contextual_help(command, context)
    
    def should_show_tip(self, context: str) -> bool:
        """Determine if a tip should be shown in current context."""
        
        profile = self.context.user_profile
        if not profile or not profile.show_tips:
            return False
        
        # Show tips more frequently for beginners
        if profile.experience_level == "beginner":
            return True
        
        # Show tips occasionally for others
        return context in ["first_time", "error_recovery", "new_feature"]