# Create missing directories
mkdir -p src/synthfuse/sigils
mkdir -p src/synthfuse/ingest
mkdir -p src/synthfuse/cabinet/roles

# Create __init__.py files
touch src/synthfuse/sigils/__init__.py
touch src/synthfuse/ingest/__init__.py
touch src/synthfuse/cabinet/roles/__init__.py

# Create minimal sigils/compiler.py
cat > src/synthfuse/sigils/compiler.py << 'EOF'
"""Sigil compiler - Formal topological constraint compiler."""

class SigilCompiler:
    """Compile Sigils to executable code."""
    
    def __init__(self):
        self.name = "SigilCompiler v0.2.0"
    
    def compile(self, sigil: str):
        """Compile a Sigil to AST."""
        return {
            "sigil": sigil,
            "ast": f"AST for {sigil}",
            "status": "compiled"
        }
    
    def generate_jax_code(self, ast):
        """Generate JAX code from AST."""
        return f"# JAX code for {ast}"
EOF

# Create minimal ingest/manager.py
cat > src/synthfuse/ingest/manager.py << 'EOF'
"""Ingestion manager - Zeta-Vault fluid ingestion."""

class IngestionManager:
    """Manage Zeta-Vault ingestion."""
    
    def __init__(self, vault_path="./vault"):
        self.vault_path = vault_path
    
    async def ingest_file(self, filepath):
        """Ingest a file into the vault."""
        return {
            "file": str(filepath),
            "hash": "abc123",
            "status": "ingested"
        }

class ZetaProjector:
    """Project data to Zeta-Manifold."""
    
    def __init__(self):
        self.name = "ZetaProjector"
    
    async def project(self, data):
        """Project data to frequency domain."""
        return {"status": "projected", "data": data}
EOF

# Create minimal cabinet roles (as before)
cat > src/synthfuse/cabinet/roles/architect.py << 'EOF'
"""Architect role."""

class Architect:
    def __init__(self):
        self.name = "Architect"
    async def blueprint(self, strategy="W-Orion"):
        return {"strategy": strategy, "status": "blueprinted"}
EOF

# Create other minimal role files
for role in engineer librarian physician shield body jury; do
cat > src/synthfuse/cabinet/roles/${role}.py << EOF
"""${role^} role."""

class ${role^}:
    def __init__(self):
        self.name = "${role^}"
    async def execute(self):
        return {"role": self.name, "status": "executed"}
EOF
done