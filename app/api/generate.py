"""
Educational content generation API endpoints
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.startup import get_system_state
from app.models.schemas import (
    ContentGenerationRequest, GeneratedContent, ContentGenerationStatus,
    ContentType, AgeGroup, StoryGenerationRequest, VocabularyGenerationRequest,
    DialogueGenerationRequest, QuestionAnswerGenerationRequest,
    StoryContent, VocabularyContent, DialogueContent, QuestionAnswerContent
)
from app.services.safety import SafetyChecker

router = APIRouter()
logger = logging.getLogger(__name__)

# Global state for generation tracking
_generation_tasks: Dict[str, Dict] = {}

@router.post("/story", response_model=StoryContent)
async def generate_story(request: StoryGenerationRequest):
    """Generate an educational story in Toaripi"""
    
    try:
        # Validate that a model is active
        system_state = get_system_state()
        if not system_state.get("model_loaded", False):
            raise HTTPException(
                status_code=400,
                detail="No model is currently active. Please activate a model first."
            )
        
        # Validate and check content safety
        safety_checker = SafetyChecker()
        
        # Check prompt safety
        safety_score = await safety_checker.check_content_safety(request.prompt, request.prompt)
        if safety_score < settings.SAFETY_THRESHOLD:
            raise HTTPException(
                status_code=400,
                detail="Content request does not meet safety guidelines"
            )
        
        # Generate story content
        story_content = await _generate_story_content(request)
        
        # Validate generated content
        content_safety = await safety_checker.check_content_safety(
            story_content["english_text"], 
            story_content["toaripi_text"]
        )
        
        if content_safety < settings.SAFETY_THRESHOLD:
            logger.warning(f"Generated story failed safety check: {content_safety}")
            # Regenerate with stricter parameters
            story_content = await _generate_story_content(request, safe_mode=True)
        
        # Create response
        result = StoryContent(
            content_type="story",
            generated_at=datetime.utcnow(),
            prompt=request.prompt,
            age_group=request.age_group,
            toaripi_text=story_content["toaripi_text"],
            english_translation=story_content["english_text"],
            cultural_elements=story_content["cultural_elements"],
            learning_objectives=story_content["learning_objectives"],
            vocabulary_words=story_content["vocabulary"],
            comprehension_questions=story_content["questions"],
            safety_score=content_safety,
            word_count=len(story_content["toaripi_text"].split()),
            estimated_reading_time_minutes=_estimate_reading_time(story_content["toaripi_text"], request.age_group)
        )
        
        logger.info(f"Generated story for prompt: {request.prompt[:50]}...")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Story generation error: {e}")
        raise HTTPException(status_code=500, detail="Story generation failed")

@router.post("/vocabulary", response_model=VocabularyContent)
async def generate_vocabulary(request: VocabularyGenerationRequest):
    """Generate vocabulary exercises in Toaripi"""
    
    try:
        # Validate that a model is active
        system_state = get_system_state()
        if not system_state.get("model_loaded", False):
            raise HTTPException(
                status_code=400,
                detail="No model is currently active. Please activate a model first."
            )
        
        # Generate vocabulary content
        vocab_content = await _generate_vocabulary_content(request)
        
        # Validate content safety
        safety_checker = SafetyChecker()
        all_text = " ".join([word["toaripi"] + " " + word["english"] for word in vocab_content["words"]])
        safety_score = await safety_checker.check_content_safety(all_text, all_text)
        
        result = VocabularyContent(
            content_type="vocabulary",
            generated_at=datetime.utcnow(),
            topic=request.topic,
            age_group=request.age_group,
            words=vocab_content["words"],
            example_sentences=vocab_content["examples"],
            practice_exercises=vocab_content["exercises"],
            cultural_context=vocab_content["cultural_context"],
            safety_score=safety_score,
            difficulty_level=vocab_content["difficulty"],
            learning_objectives=vocab_content["learning_objectives"]
        )
        
        logger.info(f"Generated {len(vocab_content['words'])} vocabulary words for topic: {request.topic}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vocabulary generation error: {e}")
        raise HTTPException(status_code=500, detail="Vocabulary generation failed")

@router.post("/dialogue", response_model=DialogueContent)
async def generate_dialogue(request: DialogueGenerationRequest):
    """Generate educational dialogue in Toaripi"""
    
    try:
        # Validate that a model is active
        system_state = get_system_state()
        if not system_state.get("model_loaded", False):
            raise HTTPException(
                status_code=400,
                detail="No model is currently active. Please activate a model first."
            )
        
        # Generate dialogue content
        dialogue_content = await _generate_dialogue_content(request)
        
        # Validate content safety
        safety_checker = SafetyChecker()
        dialogue_text = " ".join([turn["toaripi_text"] for turn in dialogue_content["dialogue"]])
        safety_score = await safety_checker.check_content_safety(dialogue_text, dialogue_text)
        
        result = DialogueContent(
            content_type="dialogue",
            generated_at=datetime.utcnow(),
            scenario=request.scenario,
            age_group=request.age_group,
            participants=dialogue_content["participants"],
            dialogue=dialogue_content["dialogue"],
            vocabulary_focus=dialogue_content["vocabulary_focus"],
            cultural_context=dialogue_content["cultural_context"],
            practice_suggestions=dialogue_content["practice_suggestions"],
            safety_score=safety_score,
            turn_count=len(dialogue_content["dialogue"]),
            estimated_duration_minutes=dialogue_content["estimated_duration"]
        )
        
        logger.info(f"Generated dialogue for scenario: {request.scenario}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dialogue generation error: {e}")
        raise HTTPException(status_code=500, detail="Dialogue generation failed")

@router.post("/qa", response_model=QuestionAnswerContent)
async def generate_questions(request: QuestionAnswerGenerationRequest):
    """Generate Q&A content for reading comprehension"""
    
    try:
        # Validate that a model is active
        system_state = get_system_state()
        if not system_state.get("model_loaded", False):
            raise HTTPException(
                status_code=400,
                detail="No model is currently active. Please activate a model first."
            )
        
        # Generate Q&A content
        qa_content = await _generate_qa_content(request)
        
        # Validate content safety
        safety_checker = SafetyChecker()
        all_text = request.text + " " + " ".join([qa["question"] + " " + qa["answer"] for qa in qa_content["questions"]])
        safety_score = await safety_checker.check_content_safety(all_text, all_text)
        
        result = QuestionAnswerContent(
            content_type="qa",
            generated_at=datetime.utcnow(),
            source_text=request.text,
            age_group=request.age_group,
            questions=qa_content["questions"],
            comprehension_level=qa_content["comprehension_level"],
            learning_objectives=qa_content["learning_objectives"],
            answer_key=qa_content["answer_key"],
            discussion_points=qa_content["discussion_points"],
            safety_score=safety_score,
            question_count=len(qa_content["questions"]),
            estimated_completion_time_minutes=qa_content["estimated_time"]
        )
        
        logger.info(f"Generated {len(qa_content['questions'])} questions for text comprehension")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Q&A generation error: {e}")
        raise HTTPException(status_code=500, detail="Q&A generation failed")

@router.post("/generate", response_model=GeneratedContent)
async def generate_content(request: ContentGenerationRequest):
    """Generate any type of educational content (unified endpoint)"""
    
    try:
        # Route to specific generation function based on content type
        if request.content_type == ContentType.STORY:
            story_request = StoryGenerationRequest(
                prompt=request.prompt,
                age_group=request.age_group,
                length=request.parameters.get("length", "medium"),
                cultural_elements=request.parameters.get("cultural_elements", []),
                learning_objectives=request.parameters.get("learning_objectives", [])
            )
            story_result = await generate_story(story_request)
            
            return GeneratedContent(
                id=f"story_{int(datetime.utcnow().timestamp())}",
                content_type=request.content_type,
                content=story_result.dict(),
                metadata={
                    "generation_time_ms": 2300,
                    "model_used": get_system_state().get("active_model", "unknown"),
                    "safety_score": story_result.safety_score,
                    "word_count": story_result.word_count
                }
            )
            
        elif request.content_type == ContentType.VOCABULARY:
            vocab_request = VocabularyGenerationRequest(
                topic=request.prompt,
                age_group=request.age_group,
                word_count=request.parameters.get("word_count", 10),
                include_examples=request.parameters.get("include_examples", True),
                difficulty_level=request.parameters.get("difficulty_level", "beginner")
            )
            vocab_result = await generate_vocabulary(vocab_request)
            
            return GeneratedContent(
                id=f"vocab_{int(datetime.utcnow().timestamp())}",
                content_type=request.content_type,
                content=vocab_result.dict(),
                metadata={
                    "generation_time_ms": 1800,
                    "model_used": get_system_state().get("active_model", "unknown"),
                    "safety_score": vocab_result.safety_score,
                    "word_count": len(vocab_result.words)
                }
            )
            
        elif request.content_type == ContentType.DIALOGUE:
            dialogue_request = DialogueGenerationRequest(
                scenario=request.prompt,
                age_group=request.age_group,
                participant_count=request.parameters.get("participant_count", 2),
                turn_count=request.parameters.get("turn_count", 6),
                vocabulary_focus=request.parameters.get("vocabulary_focus", [])
            )
            dialogue_result = await generate_dialogue(dialogue_request)
            
            return GeneratedContent(
                id=f"dialogue_{int(datetime.utcnow().timestamp())}",
                content_type=request.content_type,
                content=dialogue_result.dict(),
                metadata={
                    "generation_time_ms": 2700,
                    "model_used": get_system_state().get("active_model", "unknown"),
                    "safety_score": dialogue_result.safety_score,
                    "turn_count": dialogue_result.turn_count
                }
            )
            
        elif request.content_type == ContentType.QA:
            qa_request = QuestionAnswerGenerationRequest(
                text=request.prompt,
                age_group=request.age_group,
                question_count=request.parameters.get("question_count", 5),
                question_types=request.parameters.get("question_types", ["comprehension", "analysis"]),
                include_answers=request.parameters.get("include_answers", True)
            )
            qa_result = await generate_questions(qa_request)
            
            return GeneratedContent(
                id=f"qa_{int(datetime.utcnow().timestamp())}",
                content_type=request.content_type,
                content=qa_result.dict(),
                metadata={
                    "generation_time_ms": 2100,
                    "model_used": get_system_state().get("active_model", "unknown"),
                    "safety_score": qa_result.safety_score,
                    "question_count": qa_result.question_count
                }
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {request.content_type}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified content generation error: {e}")
        raise HTTPException(status_code=500, detail="Content generation failed")

@router.get("/capabilities")
async def get_generation_capabilities():
    """Get information about available content generation capabilities"""
    
    try:
        system_state = get_system_state()
        active_model = system_state.get("active_model")
        
        if not active_model:
            return {
                "active_model": None,
                "capabilities": [],
                "status": "no_model_active",
                "message": "No model is currently active"
            }
        
        # Get model capabilities (would be loaded from model config in production)
        capabilities = {
            "content_types": [
                {
                    "type": "story",
                    "description": "Generate educational stories in Toaripi",
                    "parameters": ["length", "cultural_elements", "learning_objectives"],
                    "age_groups": ["primary", "secondary"],
                    "average_generation_time_ms": 2300
                },
                {
                    "type": "vocabulary",
                    "description": "Generate vocabulary lists and exercises",
                    "parameters": ["word_count", "difficulty_level", "include_examples"],
                    "age_groups": ["primary", "secondary"],
                    "average_generation_time_ms": 1800
                },
                {
                    "type": "dialogue",
                    "description": "Generate conversational dialogues",
                    "parameters": ["participant_count", "turn_count", "vocabulary_focus"],
                    "age_groups": ["primary", "secondary"],
                    "average_generation_time_ms": 2700
                },
                {
                    "type": "qa",
                    "description": "Generate reading comprehension questions",
                    "parameters": ["question_count", "question_types", "include_answers"],
                    "age_groups": ["primary", "secondary"],
                    "average_generation_time_ms": 2100
                }
            ],
            "cultural_elements": [
                "fishing_activities",
                "village_life",
                "traditional_stories",
                "family_relationships",
                "nature_connection",
                "community_events"
            ],
            "learning_objectives": [
                "vocabulary_building",
                "reading_comprehension",
                "cultural_awareness",
                "language_structure",
                "communication_skills"
            ],
            "safety_features": {
                "content_filtering": True,
                "age_appropriateness": True,
                "cultural_sensitivity": True,
                "educational_alignment": True,
                "safety_threshold": settings.SAFETY_THRESHOLD
            }
        }
        
        return {
            "active_model": active_model,
            "capabilities": capabilities,
            "status": "ready",
            "model_load_time": system_state.get("model_load_time"),
            "last_generation": None  # Would track last generation time
        }
        
    except Exception as e:
        logger.error(f"Error getting generation capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to get generation capabilities")

# Helper functions for content generation
async def _generate_story_content(request: StoryGenerationRequest, safe_mode: bool = False) -> Dict[str, Any]:
    """Generate story content using the active model"""
    
    # Simulate generation delay
    await asyncio.sleep(2.3)
    
    # In production, this would call the actual model
    # For now, return educational sample content
    
    if request.age_group == AgeGroup.PRIMARY:
        story_samples = [
            {
                "toaripi_text": "Kila uma kila ai auvera herea. Mea herea mai auvera kila kila. Auvera kabi mea mai uma herea.",
                "english_text": "The children went to catch fish. The fish were big and many. The fish helped the family eat well.",
                "cultural_elements": ["fishing_activities", "family_cooperation"],
                "vocabulary": ["kila", "auvera", "herea", "mea", "uma"],
                "learning_objectives": ["vocabulary_building", "cultural_awareness"]
            }
        ]
    else:
        story_samples = [
            {
                "toaripi_text": "Kila uma auvera herea kabi mea. Auvera kila mai herea uma kabi. Mea herea kila auvera uma kabi.",
                "english_text": "The children learned to fish from their elders. The fishing knowledge passed through generations. The community shared the wisdom of the sea.",
                "cultural_elements": ["traditional_knowledge", "intergenerational_learning"],
                "vocabulary": ["kila", "auvera", "herea", "mea", "uma", "kabi"],
                "learning_objectives": ["cultural_awareness", "reading_comprehension"]
            }
        ]
    
    story = story_samples[0]
    
    return {
        "toaripi_text": story["toaripi_text"],
        "english_text": story["english_text"],
        "cultural_elements": story["cultural_elements"],
        "learning_objectives": story["learning_objectives"],
        "vocabulary": story["vocabulary"],
        "questions": [
            "Who went fishing?",
            "What did they catch?",
            "How did the community benefit?"
        ]
    }

async def _generate_vocabulary_content(request: VocabularyGenerationRequest) -> Dict[str, Any]:
    """Generate vocabulary content"""
    
    await asyncio.sleep(1.8)
    
    # Sample vocabulary based on topic
    topic_vocab = {
        "fishing": [
            {"toaripi": "auvera", "english": "fish", "definition": "Sea creature caught for food"},
            {"toaripi": "herea", "english": "catch", "definition": "To capture fish"},
            {"toaripi": "mea", "english": "big", "definition": "Large in size"},
            {"toaripi": "kila", "english": "children", "definition": "Young people"},
            {"toaripi": "uma", "english": "go", "definition": "To move to a place"}
        ],
        "family": [
            {"toaripi": "kabi", "english": "family", "definition": "Related people living together"},
            {"toaripi": "ama", "english": "father", "definition": "Male parent"},
            {"toaripi": "ina", "english": "mother", "definition": "Female parent"},
            {"toaripi": "kila", "english": "children", "definition": "Young family members"},
            {"toaripi": "koko", "english": "grandparent", "definition": "Parent's parent"}
        ]
    }
    
    words = topic_vocab.get(request.topic.lower(), topic_vocab["fishing"])[:request.word_count]
    
    return {
        "words": words,
        "examples": [
            f"Kila uma auvera herea. (The children go to catch fish.)",
            f"Mea auvera kabi herea. (The big fish helps the family.)"
        ],
        "exercises": [
            {"type": "matching", "instruction": "Match the Toaripi word with its English meaning"},
            {"type": "fill_blank", "instruction": "Fill in the missing Toaripi word"}
        ],
        "cultural_context": f"These words are commonly used in {request.topic} activities in Toaripi communities.",
        "difficulty": request.difficulty_level,
        "learning_objectives": ["vocabulary_expansion", "cultural_understanding"]
    }

async def _generate_dialogue_content(request: DialogueGenerationRequest) -> Dict[str, Any]:
    """Generate dialogue content"""
    
    await asyncio.sleep(2.7)
    
    dialogue_turns = [
        {
            "speaker": "Ama (Father)",
            "toaripi_text": "Kila, uma auvera herea?",
            "english_translation": "Children, shall we go catch fish?",
            "turn_number": 1
        },
        {
            "speaker": "Kila (Child)",
            "toaripi_text": "Io, ama! Uma auvera herea.",
            "english_translation": "Yes, father! Let's go catch fish.",
            "turn_number": 2
        },
        {
            "speaker": "Ama (Father)",
            "toaripi_text": "Auvera mea herea kabi.",
            "english_translation": "We will catch big fish for the family.",
            "turn_number": 3
        }
    ]
    
    return {
        "participants": ["Ama (Father)", "Kila (Child)"],
        "dialogue": dialogue_turns[:request.turn_count],
        "vocabulary_focus": ["auvera", "herea", "kila", "ama"],
        "cultural_context": f"This dialogue shows {request.scenario} in a traditional Toaripi setting.",
        "practice_suggestions": [
            "Practice pronunciation with a partner",
            "Role-play the scenario",
            "Create variations with different family members"
        ],
        "estimated_duration": 5
    }

async def _generate_qa_content(request: QuestionAnswerGenerationRequest) -> Dict[str, Any]:
    """Generate Q&A content"""
    
    await asyncio.sleep(2.1)
    
    questions = [
        {
            "question": "What is the main activity described in the text?",
            "answer": "Fishing",
            "question_type": "comprehension",
            "difficulty": "easy"
        },
        {
            "question": "Who are the main characters?",
            "answer": "Children and their father",
            "question_type": "comprehension",
            "difficulty": "easy"
        },
        {
            "question": "Why is fishing important to the community?",
            "answer": "It provides food for families",
            "question_type": "analysis",
            "difficulty": "medium"
        }
    ]
    
    return {
        "questions": questions[:request.question_count],
        "comprehension_level": "basic" if request.age_group == AgeGroup.PRIMARY else "intermediate",
        "learning_objectives": ["reading_comprehension", "critical_thinking"],
        "answer_key": {q["question"]: q["answer"] for q in questions},
        "discussion_points": [
            "How does fishing connect the community?",
            "What other activities bring families together?",
            "How is knowledge passed between generations?"
        ],
        "estimated_time": len(questions) * 3  # 3 minutes per question
    }

def _estimate_reading_time(text: str, age_group: AgeGroup) -> int:
    """Estimate reading time based on text length and age group"""
    
    word_count = len(text.split())
    
    # Words per minute by age group
    wpm = {
        AgeGroup.PRIMARY: 80,
        AgeGroup.SECONDARY: 150
    }
    
    reading_wpm = wpm.get(age_group, 100)
    return max(1, round(word_count / reading_wpm))