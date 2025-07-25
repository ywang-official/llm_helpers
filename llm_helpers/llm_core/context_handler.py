from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Deque
from datetime import datetime

@dataclass
class DialogueTurn:
    role: str
    content: str
    timestamp: datetime
    turn_id: int
    metadata: Optional[Dict] = None

class ContextHandler:
    def __init__(self, max_history: Optional[int] = None):
        self.max_history = max_history
        self.dialogue_history: Deque[DialogueTurn] = deque(
            maxlen=self.max_history if self.max_history else 100
        )
        self.current_turn_id = 0
        
    def add_to_history(
        self, 
        role: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> int:
        """添加一条对话记录，返回turn_id"""
        turn = DialogueTurn(
            role=role,
            content=content,
            timestamp=datetime.now(),
            turn_id=self.current_turn_id,
            metadata=metadata
        )
        self.dialogue_history.append(turn)
        self.current_turn_id += 1
        return turn.turn_id
        
    def get_history(
        self,
        last_n_turns: Optional[int] = None,
        start_turn: Optional[int] = None,
        end_turn: Optional[int] = None,
        include_metadata: bool = False
    ) -> List[Dict[str, str]]:
        """获取历史记录，支持多种过滤方式"""
        history = list(self.dialogue_history)
        
        # 按turn_id过滤
        if start_turn is not None and end_turn is not None:
            history = [
                turn for turn in history 
                if start_turn <= turn.turn_id <= end_turn
            ]
        # 获取最近n轮对话
        elif last_n_turns is not None:
            history = history[-last_n_turns:]
            
        # 转换为Claude API格式
        formatted_history = []
        for turn in history:
            message = {
                "role": turn.role,
                "content": turn.content
            }
            if include_metadata and turn.metadata:
                message["metadata"] = turn.metadata
            formatted_history.append(message)
            
        return formatted_history

    def clear_history(self) -> None:
        """清空对话历史"""
        self.dialogue_history.clear()
        
    def remove_turns(self, start_turn: int, end_turn: int) -> None:
        """删除指定范围内的对话记录"""
        self.dialogue_history = deque(
            [turn for turn in self.dialogue_history 
             if turn.turn_id < start_turn or turn.turn_id > end_turn],
            maxlen=self.max_history if self.max_history else 100
        )
        
    def get_turn_by_id(self, turn_id: int) -> Optional[DialogueTurn]:
        """根据turn_id获取特定的对话记录"""
        for turn in self.dialogue_history:
            if turn.turn_id == turn_id:
                return turn
        return None
