# infrastructure/repositories.py
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    def get_supervisor_info(self) -> str:
        pass
