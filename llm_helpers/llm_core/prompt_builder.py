from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class PromptBuilder(ABC):
    """提示词构建抽象基类"""
    
    @abstractmethod
    def load_prompts(self) -> None:
        """从YAML文件加载提示词模板"""
        pass
    
    @abstractmethod
    def build_system_prompt(self) -> str:
        """构建system prompt"""
        pass
    
    @abstractmethod
    def build_messages(self, context: Dict) -> List[Dict[str, str]]:
        """构建messages数组"""
        pass
    
    async def send_request(
        self,
        llm_handler,
        context: Dict,
        max_tokens: Optional[int] = None
    ) -> str:
        """构建完整请求并发送"""
        system = self.build_system_prompt()
        messages = self.build_messages(context)
        
        return await llm_handler.send_request(
            messages=messages,
            system=system,
            max_tokens=max_tokens
        ) 