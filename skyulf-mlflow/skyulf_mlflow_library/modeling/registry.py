"""
Model Registry for Skyulf MLflow.

This module provides a model registry for saving, loading, and managing
machine learning models with versioning and metadata tracking.
"""

import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd


class ModelRegistry:
    """
    Registry for managing machine learning models with versioning.
    
    The registry stores models with metadata in a SQLite database and
    saves model artifacts to disk. Supports versioning, tagging, and
    model lifecycle management.
    
    Parameters
    ----------
    registry_path : str or Path, optional
        Path to the registry directory. If None, uses './model_registry'.
    
    Examples
    --------
    >>> from skyulf_mlflow_library.modeling import ModelRegistry
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> import numpy as np
    >>> 
    >>> # Initialize registry
    >>> registry = ModelRegistry('./my_models')
    >>> 
    >>> # Train a model
    >>> model = RandomForestClassifier(n_estimators=10, random_state=42)
    >>> X = np.random.rand(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> model.fit(X, y)
    >>> 
    >>> # Save model
    >>> model_id = registry.save_model(
    ...     model=model,
    ...     name='my_classifier',
    ...     problem_type='classification',
    ...     metrics={'accuracy': 0.95},
    ...     tags=['production']
    ... )
    >>> 
    >>> # Load model
    >>> loaded_model = registry.load_model('my_classifier')
    >>> 
    >>> # List all models
    >>> models = registry.list_models()
    """
    
    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        """
        Initialize the model registry.
        
        Parameters
        ----------
        registry_path : str or Path, optional
            Path to the registry directory.
        """
        self.registry_path = Path(registry_path or './model_registry')
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create models directory
        self.models_path = self.registry_path / 'models'
        self.models_path.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.registry_path / 'registry.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                problem_type TEXT,
                description TEXT,
                file_path TEXT NOT NULL,
                metrics TEXT,
                tags TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(name, version)
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_model_name 
            ON models(name)
        ''')
        
        conn.commit()
        conn.close()
    
    def save_model(
        self,
        model: Any,
        name: str,
        problem_type: Optional[str] = None,
        description: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        version: Optional[int] = None,
        save_format: str = 'joblib'
    ) -> int:
        """
        Save a model to the registry.
        
        Parameters
        ----------
        model : object
            The trained model to save.
        name : str
            Name of the model.
        problem_type : str, optional
            Type of problem ('classification' or 'regression').
        description : str, optional
            Description of the model.
        metrics : dict, optional
            Performance metrics.
        tags : list of str, optional
            Tags for the model.
        version : int, optional
            Version number. If None, auto-increments.
        save_format : str, default='joblib'
            Format to save model ('joblib' or 'pickle').
        
        Returns
        -------
        int
            The model ID.
        """
        # Get next version if not specified
        if version is None:
            version = self._get_next_version(name)
        
        # Create model directory
        model_dir = self.models_path / name / f'v{version}'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        if save_format == 'joblib':
            model_file = model_dir / 'model.joblib'
            joblib.dump(model, model_file)
        elif save_format == 'pickle':
            model_file = model_dir / 'model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported save_format: {save_format}")
        
        # Convert metrics and tags to JSON
        metrics_json = json.dumps(metrics) if metrics else None
        tags_json = json.dumps(tags) if tags else None
        
        # Save metadata to database
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO models 
            (name, version, problem_type, description, file_path, 
             metrics, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, version, problem_type, description, str(model_file),
            metrics_json, tags_json, timestamp, timestamp
        ))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return model_id
    
    def load_model(
        self,
        name: str,
        version: Optional[int] = None
    ) -> Any:
        """
        Load a model from the registry.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : int, optional
            Version to load. If None, loads latest version.
        
        Returns
        -------
        object
            The loaded model.
        
        Raises
        ------
        ValueError
            If model is not found.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if version is None:
            # Load latest version
            cursor.execute('''
                SELECT file_path FROM models
                WHERE name = ?
                ORDER BY version DESC
                LIMIT 1
            ''', (name,))
        else:
            # Load specific version
            cursor.execute('''
                SELECT file_path FROM models
                WHERE name = ? AND version = ?
            ''', (name, version))
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            raise ValueError(f"Model '{name}' version {version} not found")
        
        file_path = Path(result[0])
        
        # Load model based on extension
        if file_path.suffix == '.joblib':
            return joblib.load(file_path)
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported model file format: {file_path.suffix}")
    
    def list_models(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        List all models in the registry.
        
        Parameters
        ----------
        name : str, optional
            Filter by model name.
        tags : list of str, optional
            Filter by tags (any match).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with model information.
        """
        conn = sqlite3.connect(str(self.db_path))
        
        query = 'SELECT * FROM models'
        conditions = []
        params = []
        
        if name:
            conditions.append('name = ?')
            params.append(name)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY name, version DESC'
        
        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()
        
        # Parse JSON columns
        if not df.empty:
            df['metrics'] = df['metrics'].apply(
                lambda x: json.loads(x) if x else None
            )
            df['tags'] = df['tags'].apply(
                lambda x: json.loads(x) if x else None
            )
            
            # Filter by tags if specified
            if tags:
                df = df[df['tags'].apply(
                    lambda x: x is not None and any(tag in x for tag in tags)
                )]
        
        return df
    
    def get_model_info(
        self,
        name: str,
        version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get metadata for a specific model.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : int, optional
            Version number. If None, gets latest version.
        
        Returns
        -------
        dict
            Model metadata.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if version is None:
            cursor.execute('''
                SELECT * FROM models
                WHERE name = ?
                ORDER BY version DESC
                LIMIT 1
            ''', (name,))
        else:
            cursor.execute('''
                SELECT * FROM models
                WHERE name = ? AND version = ?
            ''', (name, version))
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            raise ValueError(f"Model '{name}' version {version} not found")
        
        # Convert to dictionary
        columns = [
            'id', 'name', 'version', 'problem_type', 'description',
            'file_path', 'metrics', 'tags', 'created_at', 'updated_at'
        ]
        info = dict(zip(columns, result))
        
        # Parse JSON fields
        info['metrics'] = json.loads(info['metrics']) if info['metrics'] else None
        info['tags'] = json.loads(info['tags']) if info['tags'] else None
        
        return info
    
    def delete_model(
        self,
        name: str,
        version: Optional[int] = None
    ):
        """
        Delete a model from the registry.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : int, optional
            Version to delete. If None, deletes all versions.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        if version is None:
            # Delete all versions
            cursor.execute('SELECT file_path FROM models WHERE name = ?', (name,))
            files = cursor.fetchall()
            
            cursor.execute('DELETE FROM models WHERE name = ?', (name,))
        else:
            # Delete specific version
            cursor.execute(
                'SELECT file_path FROM models WHERE name = ? AND version = ?',
                (name, version)
            )
            files = cursor.fetchall()
            
            cursor.execute(
                'DELETE FROM models WHERE name = ? AND version = ?',
                (name, version)
            )
        
        conn.commit()
        conn.close()
        
        # Delete files
        for (file_path,) in files:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                # Try to remove parent directories if empty
                try:
                    path.parent.rmdir()
                    path.parent.parent.rmdir()
                except OSError:
                    pass
    
    def update_tags(
        self,
        name: str,
        version: int,
        tags: List[str]
    ):
        """
        Update tags for a model.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : int
            Version number.
        tags : list of str
            New tags.
        """
        tags_json = json.dumps(tags)
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE models
            SET tags = ?, updated_at = ?
            WHERE name = ? AND version = ?
        ''', (tags_json, timestamp, name, version))
        
        conn.commit()
        conn.close()
    
    def _get_next_version(self, name: str) -> int:
        """Get the next version number for a model."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT MAX(version) FROM models WHERE name = ?',
            (name,)
        )
        result = cursor.fetchone()
        conn.close()
        
        max_version = result[0] if result[0] is not None else 0
        return max_version + 1
