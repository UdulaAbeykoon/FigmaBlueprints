"""
Learn Mode Routes - Lesson Planner and Vision Verifier

This module provides endpoints for the "Learn Mode" feature that teaches users
how to manually recreate generated designs in Figma, step-by-step.
"""

import json
import base64
import httpx
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config import GROQ_API_KEY, GEMINI_API_KEY

router = APIRouter(prefix="/api/learn", tags=["learn-mode"])


# ============================================================================
# Pydantic Models
# ============================================================================

class LessonStep(BaseModel):
    id: int
    instruction: str
    success_criteria: str


class LessonPlan(BaseModel):
    steps: List[LessonStep]
    total_steps: int
    estimated_time_minutes: int


class GenerateLessonRequest(BaseModel):
    html_code: str
    framework: str = "tailwind"  # tailwind, bootstrap, etc.


class VerifyProgressRequest(BaseModel):
    current_step: LessonStep
    screenshot_base64: str  # Base64 encoded image without data:image prefix


class VerifyProgressResponse(BaseModel):
    completed: bool
    feedback: str
    confidence: float  # 0-1 confidence score


# ============================================================================
# Groq API Client
# ============================================================================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


async def call_groq_text(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 4096,
    temperature: float = 0.3
) -> str:
    """Call Groq API for text generation (Llama 3 70B)."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Add it to backend/.env"
        )
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(GROQ_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq API error: {response.text}"
            )
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


async def call_groq_vision(
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
    model: str = "llama-3.2-90b-vision-preview",
    max_tokens: int = 1024,
    temperature: float = 0.2
) -> str:
    """Call Groq Vision API for image analysis (Llama 3.2 Vision)."""
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Add it to backend/.env"
        )
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Ensure proper base64 format for JPEG (matching frontend)
    if not image_base64.startswith("data:"):
        image_base64 = f"data:image/jpeg;base64,{image_base64}"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64}
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GROQ_API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq Vision API error: {response.text}"
            )
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


# ============================================================================
# Lesson Planner Endpoint
# ============================================================================

LESSON_PLANNER_SYSTEM_PROMPT = """You are an expert Figma design instructor. Your task is to analyze HTML/CSS code and create a detailed step-by-step lesson plan for recreating the design in Figma.

CRITICAL INSTRUCTION REQUIREMENTS:
1. **Break Down Complex Shapes**: Avoid "walls of text". Split actions into small, single steps.
   - BAD: "Draw a triangle, set vertices to (0,0), (10,0)... and fill nicely."
   - GOOD Step 1: "Select the Polygon Tool and draw a generic triangle."
   - GOOD Step 2: "Set the Fill color to #6a2c70."
   - GOOD Step 3: "Double-click the triangle to enter Vector Edit mode."
   - GOOD Step 4: "Drag the top point to the bottom-left corner to match the design."
2. **Explain "How"**: When asking to change vertices/points, explain the mechanism:
   - "Double-click to edit vertices (Vector Network)."
   - "Use the Pen Tool (P) and click point-by-point..."
3. **Translate CSS to Figma Tools**: Do NOT literally translate CSS hacks. 
   - If you see `width: 0; height: 0; border: ...` (CSS Triangles), instruct to use the **Polygon Tool** or **Pen Tool**.
   - If you see `display: flex`, instruct to use **Auto Layout** (Shield+A).
4. **Be Specific about POSITION**: Always state WHERE to place an element (e.g., "at the top of the frame", "centered horizontally").
5. **Be Specific about DIMENSIONS**: Always state the size relative to the parent (e.g., "full width (100%)", "same height as the header").
6. **Be Specific about HIERARCHY**: State clearly which frame/container creates the context.
7. Use Figma-specific terminology: Frame, Rectangle, Auto Layout, Fill, Stroke, Effects, Alignment.

OUTPUT FORMAT (JSON):
{
  "steps": [
    {
      "id": 1,
      "instruction": "Detailed instruction using Figma native tools broken down into simple actions.",
      "success_criteria": "Visual confirmation."
    }
  ],
  "total_steps": <number>,
  "estimated_time_minutes": <number>
}

EXAMPLE STEPS:
- "Select the Frame tool (F) and draw a desktop frame 1440x900 pixels. Name it 'Main Container'."
- "Select the 'Polygon Tool' and draw a triangle 20x20px. Rotate it 90 degrees."
- "Double-click the triangle to edit its points. Drag the top vertex to align with the left edge."
- "Select the 'Main Container'. Draw a rectangle INSIDE it at the very top. Set its width to 1440px (Full Width)."
"""

LESSON_PLANNER_USER_PROMPT = """Analyze this {framework} code and create a Figma construction lesson plan.

Focus on the VISUAL STYLE - shapes, colors, rounded corners, and layout.
Ignore: text content (focus on the container/text box styling).

CODE:
```html
{html_code}
```

Return a JSON lesson plan with atomic steps to recreate this layout and style in Figma."""


@router.post("/generate-lesson-plan", response_model=LessonPlan)
async def generate_lesson_plan(request: GenerateLessonRequest):
    """
    Generate a step-by-step Figma lesson plan from HTML/CSS code.
    
    This endpoint analyzes the generated code and breaks it down into
    atomic construction steps that teach the user how to recreate
    the design manually in Figma.
    """
    try:
        user_prompt = LESSON_PLANNER_USER_PROMPT.format(
            framework=request.framework,
            html_code=request.html_code[:8000]  # Limit code length
        )
        
        response_text = await call_groq_text(
            system_prompt=LESSON_PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3
        )
        
        # Parse the JSON response
        try:
            lesson_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                lesson_data = json.loads(json_match.group())
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse lesson plan response"
                )
        
        # Validate and normalize the response
        steps = lesson_data.get("steps", [])
        if not steps:
            raise HTTPException(
                status_code=500,
                detail="No steps generated in lesson plan"
            )
        
        return LessonPlan(
            steps=[LessonStep(**step) for step in steps],
            total_steps=len(steps),
            estimated_time_minutes=lesson_data.get("estimated_time_minutes", len(steps) * 1)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating lesson plan: {str(e)}"
        )


# ============================================================================
# Vision Verifier Endpoint
# ============================================================================

VISION_VERIFIER_SYSTEM_PROMPT = """You are a Figma Tutor analyzing a student's screen to verify if they completed a design step.

IMPORTANT:
1. **CHECK POSITION**: Is the element in the correct place relative to its parent? (e.g., "centered", "top-left").
   - If it's floating in the middle but should be at the top, fail the step and say "Move it to the top".
2. **CHECK SHAPE/STYLE**: Does it match the visual goal?
   - If user made a rectangle instead of a triangle, fail it.
3. Be specific in feedback. "It looks like your shape is too far right" or "Use the Polygon tool for triangles".
4. IGNORE text content differences.

Your response MUST be valid JSON:
{
  "completed": true/false,
  "feedback": "Strict feedback on Position and Geometry. If mostly correct: 'Well done!'. If wrong pos: 'Move it [direction]'.",
  "confidence": 0.0-1.0
}
"""


# Global variable to cache the working model
CURRENT_VISION_MODEL = None

CANDIDATE_VISION_MODELS = [
    "llama-3.2-90b-vision-preview",
    "llama-3.2-11b-vision-preview",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    # Add potential future aliases
    "llama-3.2-vision-preview"
]

async def call_groq_vision_with_fallback(
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
    max_tokens: int = 1024,
    temperature: float = 0.2
) -> str:
    """
    Call Groq Vision API with automatic fallback for decommissioned models.
    Iterates through CANDIDATE_VISION_MODELS until one works.
    """
    global CURRENT_VISION_MODEL
    
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Add it to backend/.env"
        )
    
    # Ensure proper base64 format for JPEG (matching frontend)
    if not image_base64.startswith("data:"):
        image_base64 = f"data:image/jpeg;base64,{image_base64}"
    
    # Check if we have a cached working model
    models_to_try = []
    if CURRENT_VISION_MODEL:
        models_to_try.append(CURRENT_VISION_MODEL)
    
    # Add remaining candidates (excluding the cached one if present)
    for m in CANDIDATE_VISION_MODELS:
        if m != CURRENT_VISION_MODEL:
            models_to_try.append(m)
            
    last_error = None
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in models_to_try:
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_base64}
                                }
                            ]
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                print(f"Trying Groq Vision model: {model}...")
                response = await client.post(GROQ_API_URL, headers=headers, json=payload)
                
                if response.status_code == 200:
                    # Success! Cache this model
                    CURRENT_VISION_MODEL = model
                    print(f"Success with model: {model}")
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                
                # If error, check if it's a model issue (400/404)
                error_text = response.text
                if response.status_code in [400, 404] and ("decommissioned" in error_text or "model_not_found" in error_text or "does not exist" in error_text):
                    print(f"Model {model} failed (decommissioned/not found). Trying next...")
                    last_error = f"Model {model} decommissioned."
                    continue # Try next model
                
                # Other errors (401, 500, Rate Limit) should probably be raised immediately
                # But rate limit (429) might be specific to a model tier? Unlikely on Groq (global rate limit).
                # Let's raise for non-model errors
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Groq API error ({model}): {error_text}"
                )

            except HTTPException:
                raise
            except Exception as e:
                print(f"Error calling model {model}: {e}")
                last_error = str(e)
                continue
                
    # If all failed
    raise HTTPException(
        status_code=500,
        detail=f"All vision models failed. Last error: {last_error}"
    )


@router.post("/verify-progress", response_model=VerifyProgressResponse)
async def verify_progress(request: VerifyProgressRequest):
    """
    Verify if the user has completed the current Figma design step.
    
    Uses Groq Vision (iterating through available models) to analyze user's screen.
    """
    try:
        user_prompt = f"""CURRENT GOAL: {request.current_step.instruction}

SUCCESS CRITERIA: {request.current_step.success_criteria}

Analyze the screenshot and determine if the user has completed this step.
Check for:
- Correct sizes/proportions
- Correct colors/fills
- Correct border-radius/rounding
- Correct layout/position

Return JSON with: completed (boolean), feedback (string), confidence (0-1)"""

        # Use Dynamic Groq Vision with fallback
        response_text = await call_groq_vision_with_fallback(
            system_prompt=VISION_VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_base64=request.screenshot_base64
        )
        
        # Parse the JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*?\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                return VerifyProgressResponse(
                    completed=False,
                    feedback="Unable to analyze the screen (JSON parse error). Please try again.",
                    confidence=0.0
                )
        
        return VerifyProgressResponse(
            completed=result.get("completed", False),
            feedback=result.get("feedback", "Keep going!"),
            confidence=result.get("confidence", 0.5)
        )
        
    except HTTPException as e:
        return VerifyProgressResponse(
            completed=False,
            feedback=f"API Error: {e.detail}",
            confidence=0.0
        )
    except Exception as e:
        return VerifyProgressResponse(
            completed=False,
            feedback=f"Vision analysis unavailable. ({str(e)[:50]})",
            confidence=0.0
        )


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def learn_mode_health():
    """Check if Learn Mode is properly configured."""
    return {
        "status": "ok",
        "groq_configured": GROQ_API_KEY is not None,
        "gemini_configured": GEMINI_API_KEY is not None
    }
