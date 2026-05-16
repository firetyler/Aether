# 🧠 Java Backend with Python AI and PostgreSQL

🚧 **This project is currently under active development. Features, structure, and behavior may change frequently.** 🚧

This is a full-stack backend project developed in **Java (Spring Boot)** with an integrated **AI module written in Python**, and **PostgreSQL** as the database. The system is designed for intelligent data processing and decision-making.

---

## 🚀 Features

- REST API built with Spring Boot
- AI functionality provided by a separate Python service
- PostgreSQL for persistent data storage
- Security support using Spring Security
- Automatic table creation/update via JPA/Hibernate
- Integration between Java backend and Python AI service using HTTP communication
- AI-based memory management and natural language understanding

---

## 📦 Technologies

- Java 21
- Spring Boot 3.x
- Spring Data JPA
- PostgreSQL
- Python 3.x (for AI)
- REST API communication between Java and Python

---

## 🧠 AI Module (Python)

This project includes a **Python-based AI module** used for:
- Data analysis
- Classification
- Predictions
- Intelligent decision support

The AI service runs independently and communicates with the Java backend over HTTP. This architecture allows for modular development and easy expansion of AI capabilities.

### Python Setup (Aether AI)

1. From the repository root run the setup script to create a virtual environment and install Python dependencies.

- Windows (PowerShell):

```powershell
cd path\to\Aether
scripts\setup_env.ps1
```

- Unix / macOS:

```bash
cd path/to/Aether
./scripts/setup_env.sh
```

2. Activate the venv and run the AI service:

PowerShell:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m ai_Model.aether2
```

Unix / macOS:

```bash
source .venv/bin/activate
python -m ai_Model.aether2
```
### Java Setup
  git clone https://github.com/yourusername/aether-backend.git
  cd aether-backend
1. ** Install required dependencies
  mvn clean install

  
2.** onfigure your application.properties in java and AI model
spring.datasource.url=jdbc:postgresql://localhost:5432/aether_db
spring.datasource.username=postgres
spring.datasource.password=your_password
spring.jpa.hibernate.ddl-auto=update
spring.jpa.database-platform=org.hibernate.dialect.PostgreSQLDialect
Set youserName and password for Enpoint 

3. **Run the Java backend:
   mvn spring-boot:run

## Database 
PostgresSQL 17 or Newer are needed 
CREATE DATABASE aether_db;

##Running the Entire System
1. ** Activate the virtual environment: .venv\Scripts\activate
2. ** Run the AI service: python aether.py
3. ** Start the Java backend: mvn spring-boot:run
