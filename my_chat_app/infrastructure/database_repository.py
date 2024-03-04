# infrastructure/database_repository.py
import os
import psycopg2
from .repositories import Repository

class DatabaseRepository(Repository):
    def __init__(self):
        self._connection = psycopg2.connect(
            user=os.environ.get('postgres'),
            password=os.environ.get(''),
            host=os.environ.get('localhost'),
            database=os.environ.get('nuevabdreciente')
        )


    def get_supervisor_info(self, employee_name: str) -> str:
        query = """
            SELECT s.name 
            FROM empleados e 
            JOIN supervisores s ON e.supervisor_id = s.id 
            WHERE e.name = %s;
        """
        with self._connection.cursor() as cursor:
            cursor.execute(query, (employee_name,))
            supervisor_name = cursor.fetchone()[0]
        return supervisor_name

