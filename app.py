from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from crewai.tasks.task_output import TaskOutput
from dotenv import load_dotenv
import requests, os

# Load environment variables from .env file
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4-turbo-preview")

# Job search logic — now accepts named keyword arguments
def _search_jobs(role: str, location: str, num_results: int) -> str:
    """Search for job listings using the Adzuna API."""
    app_id = os.getenv('ADZUNA_APP_ID')
    api_key = os.getenv('ADZUNA_API_KEY')
    if not app_id or not api_key:
        return "Missing ADZUNA_APP_ID or ADZUNA_API_KEY environment variables."

    url = f"http://api.adzuna.com/v1/api/jobs/us/search/1"
    url += f"?app_id={app_id}&app_key={api_key}"
    url += f"&results_per_page={num_results}&what={role}&where={location}&content-type=application/json"

    try:
        response = requests.get(url)
        response.raise_for_status()
        jobs_data = response.json()

        job_listings = []
        for job in jobs_data.get('results', []):
            job_details = f"Title: {job['title']}, Company: {job['company']['display_name']}, Location: {job['location']['display_name']}, Description: {job['description'][:100]}..."
            job_listings.append(job_details)
        return '\n'.join(job_listings) or "No jobs found."
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"

# StructuredTool now matches keyword args
search_jobs_tool = StructuredTool.from_function(
    func=_search_jobs,
    name="Job Search Tool",
    description="Search for jobs using Adzuna. Provide 'role', 'location', and 'num_results' as input."
)

# Callback to log task results
def callback_function(output: TaskOutput):
    with open("task_output.txt", "a") as file:
        file.write(f"{output.result}\n\n")
    print("Result saved to task_output.txt")

# Agents
job_searcher_agent = Agent(
    role='Job Searcher',
    goal='Find job opportunities and report top listings',
    backstory='You are an expert in job hunting, researching actively for roles across the web.',
    llm=llm,
    tools=[search_jobs_tool],
    verbose=True,
    allow_delegation=True,
)

skills_development_agent = Agent(
    role='Skills Development Advisor',
    goal='Identify and recommend learning resources for top job-required skills',
    backstory='You help job seekers understand and build critical competencies for success.',
    llm=llm,
    verbose=True,
    allow_delegation=True,
)

interview_preparation_coach = Agent(
    role='Interview Coach',
    goal='Prepare job seekers with practice questions and personalized feedback',
    backstory='You are a communications expert helping people succeed in interviews.',
    llm=llm,
    verbose=True,
    allow_delegation=True,
)

career_advisor = Agent(
    role='Career Advisor',
    goal='Support resume optimization, LinkedIn strategies, and networking advice',
    backstory='You guide job seekers in positioning themselves effectively to land roles.',
    llm=llm,
    verbose=True,
    allow_delegation=True,
)

# Tasks
job_search_task = Task(
    description="Search for current job openings for the Senior Business Analyst role in New York using the Job Search Tool. Find 10 vacant positions in total.",
    expected_output="A list of 10 job listings including title, company, location, and short description.",
    agent=job_searcher_agent,
    tools=[search_jobs_tool],
    callback=callback_function
)

skills_highlighting_task = Task(
    description="List key skills for each job found and recommend ways to learn or improve them.",
    expected_output="A breakdown of key skills for each job, with learning resources or suggestions.",
    agent=skills_development_agent,
    context=[job_search_task],
    callback=callback_function
)

interview_preparation_task = Task(
    description="Offer mock interview questions and feedback for each of the roles found.",
    expected_output="A list of 3–5 tailored interview questions per role with tips for answering them.",
    agent=interview_preparation_coach,
    context=[job_search_task],
    callback=callback_function
)

career_advisory_task = Task(
    description="Provide tailored resume, LinkedIn, and networking tips based on the job search results.",
    expected_output="Practical advice for improving the resume, LinkedIn profile, and networking strategy for each job role.",
    agent=career_advisor,
    context=[job_search_task],
    callback=callback_function
)

# Crew setup
job_search_crew = Crew(
    agents=[job_searcher_agent, skills_development_agent, interview_preparation_coach, career_advisor],
    tasks=[job_search_task, skills_highlighting_task, interview_preparation_task, career_advisory_task],
    process=Process.hierarchical,
    manager_llm=llm
)

# Run the crew!
crew_result = job_search_crew.kickoff()
print(crew_result)
