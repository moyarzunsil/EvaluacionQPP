import os

def ensure_dir(file_path: str, create_if_not: bool = True) -> str:
    """
    Ensures the directory exists, creates it if it doesn't.
    If file_path is a file, returns its parent directory.
    
    Args:
        file_path (str): Path to file or directory
        create_if_not (bool): Whether to create directory if it doesn't exist
        
    Returns:
        str: Path to the ensured directory
        
    Raises:
        FileNotFoundError: If directory doesn't exist and create_if_not is False
    """
    # Normalize path
    file_path = os.path.normpath(os.path.expanduser(file_path))
    
    # If it's a file path, get its directory
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
        
    # Create or check directory
    if not os.path.exists(directory):
        if create_if_not:
            try:
                os.makedirs(directory)
            except FileExistsError:
                # Handle race condition in multiprocessing
                pass
        else:
            raise FileNotFoundError(
                f"Directory {directory} doesn't exist and create_if_not=False"
            )
            
    return directory 