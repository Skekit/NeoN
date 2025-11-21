import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import re
# Глобальный счетчик ID для всех нейронов в симуляции
NEURON_ID_COUNTER = 0

class Neuron:
    def __init__(self, x, y, z, neuron_type="processing"):
        """
        neuron_type: 'input', 'output', 'processing'
        """
        global NEURON_ID_COUNTER
        self.id = NEURON_ID_COUNTER
        NEURON_ID_COUNTER += 1
        
        self.position = np.array([x, y, z], dtype=float)
        self.neuron_type = neuron_type
        
        self.connections = {}  # {id_нейрона_откуда: вес_связи}
        self.activation = 0.0
        
        # Смещения и пластичность только у обучаемых нейронов
        self.bias = 0.0 if self.neuron_type == 'input' else np.random.randn() * 0.1

    def __repr__(self):
        type_char = self.neuron_type[0].upper()
        return f"N(id={self.id}, type={type_char}, pos={self.position.round(1)}, act={self.activation:.2f})"

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def calculate_activation(self, inputs):
        if self.neuron_type == 'input':
            # Активация входных нейронов устанавливается ИЗВНЕ
            self.activation = inputs.get(self.id, 0.0)
            return

        # Для output и processing нейронов
        weighted_sum = sum(
            inputs.get(neuron_id, 0) * weight
            for neuron_id, weight in self.connections.items()
        )
        self.activation = self._sigmoid(weighted_sum + self.bias)

    def update_and_rewire(self, pre_synaptic_activations, learning_params, brain_state):
        """
        Центральная функция обучения и пластичности нейрона.
        Возвращает количество связей, которые нужно создать заново.
        """
        if self.neuron_type == 'input':
            return 0
        
        # --- 1. Синаптическая пластичность (Правило Хебба + Дофамин) ---
        post_syn_activity = self.activation
        base_rate = learning_params['base_plasticity_rate']
        dopamine_level = brain_state['dopamine']
        reward_sensitivity = learning_params['reward_sensitivity']
        learning_rate = base_rate + dopamine_level * reward_sensitivity
        
        for neuron_id, pre_syn_activity in pre_synaptic_activations.items():
            if neuron_id in self.connections:
                weight_change = learning_rate * pre_syn_activity * post_syn_activity
                self.connections[neuron_id] += weight_change

        # --- 2. Физическая пластичность (только для 'processing' нейронов) ---
        if self.neuron_type == 'processing':
            movement_factor = learning_params['movement_factor']
            all_neurons = brain_state['all_neurons']
            direction_vector = np.array([0.0, 0.0, 0.0])
            total_weight = 0.0
            
            for neuron_id, weight in self.connections.items():
                if pre_synaptic_activations.get(neuron_id, 0) > 0.1:
                    target_pos = all_neurons[neuron_id].position
                    vec_to_target = target_pos - self.position
                    direction_vector += vec_to_target * abs(weight)
                    total_weight += abs(weight)
            
            if total_weight > 0:
                self.position += (direction_vector / total_weight) * movement_factor
        
        # --- 3. Структурная пластичность (Pruning) ---
        pruning_threshold = learning_params['pruning_threshold']
        dead_connections = [nid for nid, weight in self.connections.items() if abs(weight) < pruning_threshold]
        
        for nid in dead_connections:
            del self.connections[nid]
        
        # Возвращаем количество удаленных связей, которые нужно воссоздать
        return len(dead_connections)

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch # Модели transformers работают с torch.Tensor

# ... (класс Neuron остается прежним) ...

class Brain:
    def __init__(self, genome):
        self.genome = genome
        self.neurons = {}
        self.dopamine = 0.0
        self.type_map = {'input': [], 'output': [], 'processing': []}

        # --- Инициализация NLP-компонентов ---
        print("Загрузка NLP модели и токенизатора...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.genome['tokenizer_model'])
        # Загружаем саму модель, чтобы получить ее веса
        model = AutoModel.from_pretrained(self.genome['tokenizer_model'])
        # Извлекаем матрицу эмбеддингов. .detach().numpy() конвертирует ее в NumPy массив
        self.embedding_matrix = model.get_input_embeddings().weight.detach().numpy()
        self.embedding_dim = self.embedding_matrix.shape[1] # Размер эмбеддинга, например, 768
        print(f"NLP модель загружена. Размер эмбеддинга: {self.embedding_dim}")

        self._create_neurons()
        self._create_initial_connections()
        # ... (остальные методы __init__ без изменений) ...

    def _create_neurons(self):
        global NEURON_ID_COUNTER
        NEURON_ID_COUNTER = 0 # Сбрасываем для нового мозга

        # Входные нейроны - по размеру эмбеддинга
        for i in range(self.embedding_dim):
            neuron = Neuron(x=i * 0.1, y=0, z=0, neuron_type='input')
            self.neurons[neuron.id] = neuron
            self.type_map['input'].append(neuron.id)
        
        # Выходные нейроны - по размеру эмбеддинга
        for i in range(self.embedding_dim):
            neuron = Neuron(x=i * 0.1, y=10, z=0, neuron_type='output')
            self.neurons[neuron.id] = neuron
            self.type_map['output'].append(neuron.id)
            
        # Процессинговые нейроны
        for _ in range(self.genome['neuron_count']):
            pos = np.random.rand(3) * 10
            neuron = Neuron(x=pos[0], y=pos[1], z=pos[2], neuron_type='processing')
            self.neurons[neuron.id] = neuron
            self.type_map['processing'].append(neuron.id)

    # ... (_find_neighbors, _create_initial_connections остаются прежними) ...

    def _find_neighbors(self, neuron, source_pool, count):
        """Находит `count` ближайших соседей для нейрона из пула `source_pool`."""
        # Это все еще O(N), но мы вызываем его реже.
        # Для реальной скорости здесь нужна оптимизация (k-d tree, grid hash).
        neighbors = []
        for sid in source_pool:
            source_neuron = self.neurons[sid]
            dist = np.linalg.norm(neuron.position - source_neuron.position)
            neighbors.append((dist, source_neuron))
        
        neighbors.sort(key=lambda x: x[0])
        return [n for dist, n in neighbors[:count]]

    def _create_initial_connections(self):
        # Процессинговые и выходные нейроны создают связи от входных и процессинговых
        source_pool = self.type_map['input'] + self.type_map['processing']
        
        for neuron_id in self.type_map['processing'] + self.type_map['output']:
            neuron = self.neurons[neuron_id]
            # Ищем ближайших соседей, чтобы установить начальные связи
            potential_partners = self._find_neighbors(neuron, source_pool, self.genome['initial_connection_count'] * 2)
            
            # Берем случайный набор из ближайших
            partners_to_connect = np.random.choice(
                potential_partners, 
                size=min(len(potential_partners), self.genome['initial_connection_count']),
                replace=False
            )
            for partner in partners_to_connect:
                if partner.id != neuron.id:
                    neuron.connections[partner.id] = np.random.randn() * 0.5

    def _get_output_token_id(self):
        """
        Сравнивает вектор активации выходных нейронов с матрицей эмбеддингов
        и находит самый похожий токен.
        """
        # 1. Собираем вектор активации выходного слоя
        output_vector = np.array([self.neurons[nid].activation for nid in self.type_map['output']])
        
        # 2. Вычисляем косинусное сходство между выходным вектором и всеми векторами в матрице
        # Нормализуем векторы
        output_vector_norm = np.linalg.norm(output_vector)
        if output_vector_norm == 0: return 0 # Если вектор нулевой, возвращаем "пустой" токен
        
        embedding_norms = np.linalg.norm(self.embedding_matrix, axis=1)
        
        # Вычисляем скалярное произведение
        dot_products = np.dot(self.embedding_matrix, output_vector)
        
        # Косинусное сходство
        cosine_similarities = dot_products / (embedding_norms * output_vector_norm)
        
        # 3. Находим индекс с максимальным сходством
        return np.argmax(cosine_similarities)


    def tick(self, input_token_id=None):
        # --- Шаг 1: Подготовка входов ---
        external_inputs = {}
        if input_token_id is not None:
            # Получаем вектор эмбеддинга для входного токена
            input_embedding = self.embedding_matrix[input_token_id]
            # Устанавливаем активации входных нейронов
            for i, neuron_id in enumerate(self.type_map['input']):
                external_inputs[neuron_id] = input_embedding[i]
        # Собираем все активации с прошлого шага
        last_activations = {nid: n.activation for nid, n in self.neurons.items()}
        # --- Шаг 2: Активация нейронов ---
        for neuron_id in self.type_map['input']:
            self.neurons[neuron_id].calculate_activation(external_inputs)
        for neuron_id in self.type_map['processing'] + self.type_map['output']:
            self.neurons[neuron_id].calculate_activation(last_activations)

        # --- Шаг 3: Обучение и пластичность ---
        brain_state = {
            'dopamine': self.dopamine,
            'all_neurons': self.neurons
        }
        
        for neuron_id in self.type_map['processing'] + self.type_map['output']:
            neuron = self.neurons[neuron_id]
            
            pre_synaptic_activations = {
                nid: last_activations.get(nid, 0) for nid in neuron.connections.keys()
            }
            
            rewire_count = neuron.update_and_rewire(pre_synaptic_activations, self.genome, brain_state)
            
            # --- Шаг 4: Обработка запросов на переподключение (rewiring) ---
            if rewire_count > 0:
                # Ищем новых партнеров в радиусе
                source_pool = self.type_map['input'] + self.type_map['processing']
                potential_partners = self._find_neighbors(neuron, source_pool, self.genome['rewire_candidate_count'])
                
                # Выбираем случайных из кандидатов
                if potential_partners:
                    new_partners = np.random.choice(
                        potential_partners, 
                        size=min(len(potential_partners), rewire_count),
                        replace=False
                    )
                    for partner in new_partners:
                        if partner.id != neuron.id and partner.id not in neuron.connections:
                            neuron.connections[partner.id] = np.random.randn() * 0.1
        
        self.dopamine = 0.0
        # ... (Шаги 2, 3, 4 - активация и обучение - остаются БЕЗ ИЗМЕНЕНИЙ) ...
        # В этом и прелесть - внутренняя логика мозга не меняется!

        # --- Шаг 5: Получение результата ---
        # Вместо старого кода вызываем новую функцию
        return self._get_output_token_id()


# --- Пример использования ---

# Добавляем новый ген для переподключения
default_genome = {
    'embedding_dim_for_io': 768, # Соответствует bert-base-multilingual-cased
    'neuron_count': 500,
    'initial_connection_count': 5,
    'base_plasticity_rate': 0.001,
    'reward_sensitivity': 0.05,
    'pruning_threshold': 0.05,
    'movement_factor': 0.01, 
    'rewire_candidate_count': 20, # Сколько соседей рассматривать для новой связи
    'tokenizer_model': 'bert-base-multilingual-cased'
}

import random
from tqdm import tqdm # Для красивого прогресс-бара

# ... (Код классов Neuron и Brain) ...

import requests
import json
import re

def llm_dialogue_fitness(brain, dialogue_length=5, ticks_per_thought=50):
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "llama3.1:8b" # Укажите вашу модель

    system_prompt = """
Ты — ИИ-Тьютор. Твоя задача — провести короткий диалог с развивающейся нейронной системой, "Учеником".
Твоя цель — оценить, насколько осмысленны и релевантны его ответы.
Не выводи ход своих мыслей
Твой ответ ОБЯЗАН состоять из двух частей:
1. Твоя реплика в диалоге.
2. На новой строке, JSON-объект с твоей оценкой ПОСЛЕДНЕГО ответа Ученика.
В твоей реплике НЕ должно быть переноса строки.
JSON-объект должен иметь два ключа: "dopamine" (число от -1.0 до 1.0) и "thought" (строка с объяснением).

ПРИМЕР ТВОЕГО ОТВЕТА:
Интересная мысль! А как ты думаешь, повлияет ли это на творческие профессии?
{"dopamine": 0.8, "thought": "Ученик дал релевантный ответ, но можно было бы развить мысль глубже."}

ВАЖНО: Начни диалог с простого, открытого вопроса. В своем ПЕРВОМ сообщении НЕ добавляй JSON-объект.
"""
    
    history = [{"role": "system", "content": system_prompt}]
    total_fitness_score = 0.0

    # --- Ход 0: Инициация диалога Учителем ---
    try:
        print("ИИ-Тьютор начинает диалог...")
        # Собираем историю для промта
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        payload = {"model": MODEL_NAME, "prompt": prompt_text, "stream": False}
        response = requests.post(OLLAMA_URL, json=payload)
        response_data = response.json()
        tutor_reply = response_data['response'].strip()
        
        history.append({"role": "assistant", "content": tutor_reply})
        print(f"Учитель (LLM): {tutor_reply}")
        
    except Exception as e:
        print(f"Критическая ошибка на старте диалога: {e}")
        return -100 # Возвращаем очень низкий фитнес

    # --- Основной цикл диалога ---
    for i in range(dialogue_length):
        # --- Ход Ученика ---
        student_reply = ""
        try:
            student_input_ids = brain.tokenizer.encode(tutor_reply, add_special_tokens=False)
            for token_id in student_input_ids:
                brain.tick(input_token_id=token_id)
            
            for _ in range(ticks_per_thought): brain.tick()
            
            student_output_tokens = []
            current_token = brain.tokenizer.bos_token_id
            for _ in range(30):
                current_token = brain.tick(input_token_id=current_token)
                if current_token == brain.tokenizer.eos_token_id: break
                student_output_tokens.append(current_token)
            
            student_reply = brain.tokenizer.decode(student_output_tokens, skip_special_tokens=True).strip()
            # Если ответ пустой, делаем его "молчанием", чтобы LLM могла его оценить
            if not student_reply:
                student_reply = "[молчание]"

            print(f"Ученик (Brain): {student_reply}")
            history.append({"role": "user", "content": student_reply})

        except Exception as e:
            print(f"Ошибка во время хода Ученика: {e}")
            history.append({"role": "user", "content": "[ошибка генерации]"})


        # --- Ход Учителя с оценкой ---
        try:
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            payload = {"model": MODEL_NAME, "prompt": prompt_text, "stream": False, "format": "json"}
            response = requests.post(OLLAMA_URL, json=payload)
            response_data = response.json()
            tutor_response_full = response_data['response'].strip()
            
            # Более надежный парсинг JSON
            json_part = None
            # Ищем JSON объект с помощью регулярного выражения
            match = re.search(r'\{.*\}', tutor_response_full, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)
                    json_part = json.loads(json_str)
                    # Убираем JSON из реплики
                    tutor_reply = tutor_response_full.replace(json_str, '')
                except json.JSONDecodeError:
                    tutor_reply = tutor_response_full # Если не смогли распарсить, оставляем как есть
            else:
                tutor_reply = tutor_response_full
                
            dopamine_signal = 0.0
            tutor_thought = "Оценка не найдена."
            if json_part:
                dopamine_signal = float(json_part.get('dopamine', 0.0))
                tutor_thought = json_part.get('thought', 'Мысль не найдена.')

            print(f"Учитель (LLM): {tutor_reply}")
            print(f"   -> Оценка: Дофамин={dopamine_signal:.2f} ({tutor_thought})")
            
            total_fitness_score += dopamine_signal
            brain.dopamine = dopamine_signal
            brain.tick() # Усвоение дофамина

            history.append({"role": "assistant", "content": tutor_response_full})

        except Exception as e:
            print(f"Ошибка во время хода Учителя: {e}")
            total_fitness_score -= 1 # Штраф за сбой

    return total_fitness_score

        

class GeneticAlgorithm:
    def __init__(self, population_size, genome_template, fitness_function):
        """
        Инициализация генетического алгоритма.

        Args:
            population_size (int): Количество "мозгов" в популяции.
            genome_template (dict): Шаблон генома с диапазонами для каждого гена.
            fitness_function (function): Функция, которая принимает мозг и возвращает его
                                         оценку (число).
        """
        self.population_size = population_size
        self.genome_template = genome_template
        self.fitness_function = fitness_function
        self.population = self._create_initial_population()
        self.generation_number = 0
        self.best_fitness_history = []

    def _create_random_genome(self):
        """Создает один случайный геном на основе шаблона."""
        genome = {}
        for gene, (min_val, max_val) in self.genome_template.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                genome[gene] = random.randint(min_val, max_val)
            else:
                genome[gene] = random.uniform(min_val, max_val)
        # Специальный случай для модели токенизатора
        genome['tokenizer_model'] = 'bert-base-multilingual-cased'
        return genome

    def _create_initial_population(self):
        """Создает начальную популяцию со случайными геномами."""
        population = []
        for _ in range(self.population_size):
            population.append(self._create_random_genome())
        return population

    def _crossover(self, parent1_genome, parent2_genome):
        """Одноточечное скрещивание генов двух родителей."""
        child_genome = {}
        # Выбираем случайную точку для "разреза" генома
        crossover_point = random.randint(1, len(self.genome_template) - 2)
        
        # Конвертируем словари в списки для удобства
        keys = list(self.genome_template.keys())
        
        for i, key in enumerate(keys):
            if key == 'tokenizer_model': continue # Не скрещиваем модель
            
            if i < crossover_point:
                child_genome[key] = parent1_genome[key]
            else:
                child_genome[key] = parent2_genome[key]
        
        child_genome['tokenizer_model'] = parent1_genome['tokenizer_model']
        return child_genome

    def _mutate(self, genome, mutation_rate, mutation_strength):
        """Вносит случайные изменения в гены."""
        mutated_genome = genome.copy()
        for gene, (min_val, max_val) in self.genome_template.items():
            if gene == 'tokenizer_model': continue
                
            if random.random() < mutation_rate:
                # Добавляем небольшое случайное значение
                change = (random.random() - 0.5) * 2 * mutation_strength * (max_val - min_val)
                new_value = mutated_genome[gene] + change
                
                # Убеждаемся, что не вышли за границы
                if isinstance(min_val, int):
                    new_value = int(np.clip(new_value, min_val, max_val))
                else:
                    new_value = np.clip(new_value, min_val, max_val)
                
                mutated_genome[gene] = new_value
        return mutated_genome

    def _selection(self, population_with_fitness, tournament_size=3):
        """Турнирный отбор: выбираем лучшего из случайной группы."""
        selected = []
        for _ in range(len(population_with_fitness)):
            # Выбираем случайных участников для турнира
            tournament_contenders = random.sample(population_with_fitness, tournament_size)
            # Победитель - тот, у кого фитнес выше
            winner = max(tournament_contenders, key=lambda x: x[1])
            selected.append(winner[0]) # Добавляем геном победителя
        return selected

    def run_generation(self, mutation_rate=0.1, mutation_strength=0.1, elitism_count=1):
        """Запускает один полный цикл эволюции: оценка, отбор, скрещивание, мутация."""
        self.generation_number += 1
        print(f"\n--- Поколение {self.generation_number} ---")

        # 1. Оценка приспособленности
        population_with_fitness = []
        # Используем tqdm для наглядности процесса оценки
        for genome in tqdm(self.population, desc="Оценка популяции"):
            brain = Brain(genome)
            fitness = self.fitness_function(brain)
            population_with_fitness.append((genome, fitness))

        # Сортируем по фитнесу для статистики и элитизма
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        best_genome_this_gen = population_with_fitness[0][0]
        best_fitness_this_gen = population_with_fitness[0][1]
        self.best_fitness_history.append(best_fitness_this_gen)
        
        avg_fitness = np.mean([f for g, f in population_with_fitness])
        print(f"Лучший фитнес: {best_fitness_this_gen:.4f}, Средний фитнес: {avg_fitness:.4f}")
        print(f"Лучший геном: { {k: round(v, 4) if isinstance(v, float) else v for k, v in best_genome_this_gen.items() if k != 'tokenizer_model'} }")

        # 2. Отбор
        selected_parents = self._selection(population_with_fitness, tournament_size=3)
        
        # 3. Скрещивание и Мутация
        next_population = []
        
        # Элитизм: лучшие особи переходят в следующее поколение без изменений
        for i in range(elitism_count):
            next_population.append(population_with_fitness[i][0])

        # Создаем остальных потомков
        while len(next_population) < self.population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = self._crossover(parent1, parent2)
            mutated_child = self._mutate(child, mutation_rate, mutation_strength)
            next_population.append(mutated_child)
            
        self.population = next_population

# --- Пример использования: Создаем "школу" и запускаем эволюцию ---

def simple_arithmetic_fitness(brain, num_tests=10):
    """
    Простая фитнес-функция для оценки решения примеров типа "2+3=5".
    """
    total_score = 0
    for _ in range(num_tests):
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        correct_result = a + b
        
        # Формируем входную последовательность токенов
        # Например, для "2+3" это будут ID токенов '2', '+', '3'
        input_text = f"{a}+{b}"
        input_token_ids = brain.tokenizer.encode(input_text, add_special_tokens=False)
        
        # Подаем последовательность в мозг
        for token_id in input_token_ids:
            brain.tick(token_id)
        
        # Получаем ответ от мозга (пока один токен)
        # Мозг должен сгенерировать токен для правильного ответа
        # Здесь мы упрощаем: просто смотрим на состояние после входа
        output_token_id = brain.tick() # Тик без входа для генерации
        
        # Сравниваем ответ
        try:
            # Преобразуем ID токена ответа в строку
            result_str = brain.tokenizer.decode([output_token_id])
            if result_str.isdigit() and int(result_str) == correct_result:
                total_score += 1 # Награда за правильный ответ
                # Даем "дофаминовое" подкрепление (в реальной фитнес-функции это не нужно,
                # но показывает как это могло бы работать)
                brain.dopamine = 1.0 
        except:
            pass # Если декодирование не удалось, просто пропускаем

    return total_score



# Шаблон генома с диапазонами значений [min, max]
genome_template = {
    'neuron_count': [100, 500],
    'initial_connection_count': [5, 20],
    'base_plasticity_rate': [0.001, 0.01],
    'reward_sensitivity': [0.001, 0.1],
    'pruning_threshold': [0.01, 0.1],
    'movement_factor': [0.0, 0.05],
    'rewire_candidate_count': [10, 50],
}

# Инициализируем ГА
ga = GeneticAlgorithm(
    population_size=10, # Для теста, в реальности 50-100
    genome_template=genome_template,
    fitness_function=llm_dialogue_fitness
)

# Запускаем несколько поколений эволюции
num_generations = 5 # Для теста, в реальности 100+
for i in range(num_generations):
    ga.run_generation(mutation_rate=0.15, mutation_strength=0.2, elitism_count=2)

# Можно построить график эволюции

plt.plot(ga.best_fitness_history)
plt.xlabel("Поколение")
plt.ylabel("Лучший фитнес")
plt.title("Эволюция")
plt.show()




