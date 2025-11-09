"""
Template utilities for FastAPI application.

This module provides Flask-compatible template functions and utilities
for the FastAPI application to maintain compatibility with existing templates.
"""
from pathlib import Path
from fastapi.templating import Jinja2Templates


def setup_templates(base_dir: Path) -> Jinja2Templates:
    """
    Setup Jinja2 templates with Flask-compatible functions.

    Args:
        base_dir: Base directory path for the application

    Returns:
        Configured Jinja2Templates instance
    """
    templates_dir = base_dir / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    # Add Flask compatibility functions to templates
    def url_for(name: str, **kwargs):
        """FastAPI version of Flask's url_for function"""
        # Handle any extra arguments by ignoring them
        routes = {
            'index': '/',
            'auth.login': '/login',
            'auth.logout': '/logout',
            'auth.test_users': '/admin/users',  # Admin users page
            'admin.dashboard': '/admin/dashboard',  # Admin dashboard
            'admin.users': '/admin/users',  # Admin users management
            'data.data_dashboard': '/data-ingestion',  # Data ingestion page
            'ml_workflow_page': '/ml-workflow',
            'static': '/static',
        }

        if name == 'static':
            filename = kwargs.get('filename', kwargs.get('path', ''))
            return f"/static/{filename}"

        # Return the route, ignoring any extra kwargs
        return routes.get(name, '/')

    def get_flashed_messages(**kwargs):
        """Flask compatibility - return empty list"""
        return []

    # Make Flask compatibility functions available in all templates
    templates.env.globals['url_for'] = url_for
    templates.env.globals['get_flashed_messages'] = get_flashed_messages

    return templates


def add_template_context_processors(templates: Jinja2Templates) -> None:
    """
    Add additional context processors to templates if needed.

    Args:
        templates: Jinja2Templates instance to modify
    """
    # Add any additional global template functions here
    pass
