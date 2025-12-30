import os

def create_frontend_structure():
    # Define the directory structure
    structure = {
        'career-assistant-frontend': {
            'src': {
                'components': {
                    'Layout': ['Layout.jsx', 'Sidebar.jsx', 'Navbar.jsx'],
                    'Dashboard': [],
                    'Profile': [],
                    'Common': []
                },
                'pages': {
                    'Dashboard': [],
                    'Profile': [],
                    'Jobs': []
                },
                'services': ['api.js', 'auth.js', 'jobs.js'],
                'contexts': ['AuthContext.jsx', 'ThemeContext.jsx'],
                'utils': ['helpers.js'],
                'styles': ['global.css']
            },
            'public': {}
        }
    }
    
    # Root files in src/
    src_root_files = ['App.jsx', 'main.jsx']
    
    # Root files in career-assistant-frontend/
    root_files = ['index.html', 'vite.config.js', 'tailwind.config.js', 
                  'postcss.config.js', 'package.json']
    
    created_count = 0
    skipped_count = 0
    
    def create_directory(path):
        nonlocal created_count, skipped_count
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"✓ Created directory: {path}")
            created_count += 1
        else:
            print(f"⊙ Skipped (exists): {path}")
            skipped_count += 1
    
    def create_file(path):
        nonlocal created_count, skipped_count
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write('')  # Create empty file
            print(f"✓ Created file: {path}")
            created_count += 1
        else:
            print(f"⊙ Skipped (exists): {path}")
            skipped_count += 1
    
    def process_structure(base_path, structure_dict):
        for key, value in structure_dict.items():
            current_path = os.path.join(base_path, key)
            
            if isinstance(value, dict):
                # It's a directory with subdirectories
                create_directory(current_path)
                process_structure(current_path, value)
            elif isinstance(value, list):
                # It's a directory with files
                create_directory(current_path)
                for file in value:
                    file_path = os.path.join(current_path, file)
                    create_file(file_path)
    
    # Create main directory
    base_dir = 'career-assistant-frontend'
    create_directory(base_dir)
    
    # Create root files
    for file in root_files:
        file_path = os.path.join(base_dir, file)
        create_file(file_path)
    
    # Process the nested structure
    process_structure(base_dir, structure['career-assistant-frontend'])
    
    # Create src root files
    src_dir = os.path.join(base_dir, 'src')
    for file in src_root_files:
        file_path = os.path.join(src_dir, file)
        create_file(file_path)
    
    print("\n" + "="*50)
    print(f"✓ Process completed!")
    print(f"✓ Created: {created_count} items")
    print(f"⊙ Skipped: {skipped_count} items (already exist)")
    print("="*50)

if __name__ == "__main__":
    create_frontend_structure()