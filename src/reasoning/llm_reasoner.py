"""
Stage 5: LLM Reasoning Module

Uses LLM to parse multi-entity boxes, assign entity types, and infer procedural relations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from typing import List, Dict, Any
import os

from ..utils.io_utils import load_json, save_json, ensure_dir


class LLMReasoner:
    """LLM-based reasoning for procedural knowledge extraction."""
    
    def __init__(self, config: Dict):
        """
        Initialize LLM reasoner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        reasoning_config = config.get('reasoning', {})
        
        self.model_name = reasoning_config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')
        self.device = reasoning_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = reasoning_config.get('max_new_tokens', 2048)
        self.temperature = reasoning_config.get('temperature', 0.1)
        self.top_p = reasoning_config.get('top_p', 0.9)
        
        # Prompt configuration
        prompt_config = reasoning_config.get('prompt', {})
        self.system_message = prompt_config.get(
            'system_message',
            "You are an expert at parsing procedural knowledge from industrial troubleshooting diagrams."
        )
        
        # Load model and tokenizer
        print(f"Loading LLM model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def _build_prompt(self, diagram_data: Dict) -> str:
        """
        Build prompt for LLM reasoning.
        
        Args:
            diagram_data: Dictionary containing elements and arrows
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are analyzing a procedural troubleshooting diagram. Your task is to:
1. Parse multi-entity boxes into individual entities based on line breaks and bullets
2. Assign entity types (Condition, Action, Decision) to each entity
3. Infer procedural relations from the arrow graph and sequential ordering

Input diagram structure:

ELEMENTS:
"""
        
        # Add elements
        for element in diagram_data['elements']:
            prompt += f"\nElement {element['id']}:\n"
            prompt += f"  Type: {element.get('type', 'unknown')}\n"
            prompt += f"  Text: {element.get('text', '')}\n"
            prompt += f"  Position: {element['bbox']}\n"
        
        # Add arrows
        prompt += "\nARROWS:\n"
        for arrow in diagram_data['arrows']:
            label_str = f" (label: {arrow['label']})" if arrow.get('label') else ""
            prompt += f"  {arrow['source']} -> {arrow['target']}{label_str}\n"
        
        prompt += """
Please output a JSON object with the following structure:
{
  "entities": [
    {
      "id": "unique_entity_id",
      "text": "entity text content",
      "type": "Condition|Action|Decision",
      "source_element_id": element_id_this_came_from
    }
  ],
  "relations": [
    {
      "source_entity_id": "entity_id",
      "target_entity_id": "entity_id",
      "relation_type": "procedural_flow",
      "label": "optional_label_like_ja_or_nee"
    }
  ]
}

Guidelines:
- For observation boxes: extract each line/bullet as a separate Condition entity
- For decision boxes: create a Decision entity
- For action boxes: extract each line/bullet as a separate Action entity
- Relations should follow the arrow graph
- Within multi-entity boxes, create sequential relations between entities
- Preserve "ja"/"nee" labels from decision branches

Output only the JSON, no additional text.
"""
        
        return prompt
    
    def _parse_llm_output(self, output_text: str) -> Dict:
        """
        Parse LLM output to extract JSON.
        
        Args:
            output_text: Raw LLM output
            
        Returns:
            Parsed dictionary
        """
        # Try to find JSON in output
        try:
            # Look for JSON block
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in output")
            
            json_str = output_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            return result
        
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            print(f"Output was: {output_text}")
            return {
                'entities': [],
                'relations': [],
                'error': str(e)
            }
    
    def reason(self, arrow_json_path: str, output_dir: str) -> Dict:
        """
        Perform LLM reasoning on diagram data.
        
        Args:
            arrow_json_path: Path to arrow detection results JSON
            output_dir: Directory to save reasoning results
            
        Returns:
            Reasoning results dictionary
        """
        # Load arrow detection results
        diagram_data = load_json(arrow_json_path)
        
        # Build prompt
        user_prompt = self._build_prompt(diagram_data)
        
        # Format messages
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate response
        print("Generating LLM response...")
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse output
        parsed_result = self._parse_llm_output(response)
        
        # Save results
        image_name = diagram_data['image_name']
        result = {
            'image_path': diagram_data['image_path'],
            'image_name': image_name,
            'entities': parsed_result.get('entities', []),
            'relations': parsed_result.get('relations', []),
            'raw_llm_output': response
        }
        
        ensure_dir(output_dir)
        output_path = os.path.join(output_dir, f"{image_name}_reasoning.json")
        save_json(result, output_path)
        
        return result
