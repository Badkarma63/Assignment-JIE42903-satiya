import csv
import random
import requests
import io
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# STEP 1: READ CSV FILE DIRECTLY FROM GITHUB
# ------------------------------------------------------------

url = "https://raw.githubusercontent.com/Badkarma63/Assignment-JIE42903-satiya/refs/heads/main/program_ratings.csv"

response = requests.get(url)
if response.status_code != 200:
    st.error("⚠️ Error loading CSV from GitHub. Please check your file link or repo settings.")
else:
    content = response.content.decode('utf-8')
    reader = csv.reader(io.StringIO(content))
    header = next(reader)

    program_ratings_dict = {}
    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings_dict[program] = ratings

    st.success("✅ CSV loaded successfully from GitHub.")
    st.write("Programs found:", list(program_ratings_dict.keys()))

# ------------------------------------------------------------
# STEP 2: DEFINE PARAMETERS AND FUNCTIONS
# ------------------------------------------------------------

ratings = program_ratings_dict
GEN = 100
POP = 50
EL_S = 2
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))  # 6 AM – 11 PM

# Fitness Function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if time_slot < len(ratings[program]):
            total_rating += ratings[program][time_slot]
    return total_rating

# Initialize Population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules

# Find Best Schedule (Brute Force)
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
    return best_schedule

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Genetic Algorithm
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=0.8, mutation_rate=0.2, elitism_size=EL_S):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    # Store GA stats for table
    ga_stats = []

    for generation in range(generations):
        new_population = []
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        best_fit = fitness_function(population[0])
        avg_fit = sum(fitness_function(s) for s in population) / len(population)
        ga_stats.append({"Generation": generation+1, "Best Fitness": best_fit, "Average Fitness": round(avg_fit,2)})

        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0], ga_stats  # return final schedule + GA stats

# ------------------------------------------------------------
# STEP 3: STREAMLIT INTERFACE
# ------------------------------------------------------------

st.title("📺 Scheduling Using Genetic Algorithm")
st.markdown("**Course Code:** JIE42903 – Evolutionary Computing")
st.markdown("This app generates an optimal schedule using a Genetic Algorithm (GA).")

# Sliders for user input
CO_R = st.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8)
MUT_R = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02)

# Run button
if st.button("🚀 Run Genetic Algorithm"):
    st.write("Running GA with the following parameters:")
    st.write(f"- Crossover Rate: {CO_R}")
    st.write(f"- Mutation Rate: {MUT_R}")

    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules)

    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
    genetic_schedule, ga_stats = genetic_algorithm(initial_best_schedule,
                                               generations=GEN,
                                               population_size=POP,
                                               crossover_rate=CO_R,
                                               mutation_rate=MUT_R,
                                               elitism_size=EL_S)

    final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

    schedule_df = pd.DataFrame({
        "Time Slot": [f"{slot:02d}:00" for slot in all_time_slots[:len(final_schedule)]],
        "Program": final_schedule
    })

    st.subheader("📊 GA Progress per Generation")
   st.table(pd.DataFrame(ga_stats))
    st.success(f"⭐ Total Ratings: {fitness_function(final_schedule)}")

    st.caption("Note: Results may vary slightly due to GA randomness.")



