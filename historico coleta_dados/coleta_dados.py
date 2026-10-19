"""
Coleta dados agrupados por FASE CONCLUÍDA (GA e GS).
Gera um CSV formatado para treinamento de modelos de Machine Learning (RL / Regressão).
"""

import os
import sys
import traci
import pandas as pd

# ==================== CONFIGURAÇÕES ====================
SUMO_BINARY = "sumo"              # "sumo-gui" para visualizar
CONFIG_FILE = "teste.sumocfg"     # Ajustado para o seu arquivo
OUTPUT_CSV = "dataset_ml_semaforo.csv"

TEMPO_MAXIMO = 3600               # 1 hora de simulação para gerar bastante dado
TLS_ID = "J2"                     # ID do seu semáforo
# =======================================================

def get_arestas_chegada(tls_id):
    """Mapeia as arestas (edges) que chegam ao semáforo"""
    links = traci.trafficlight.getControlledLinks(tls_id)
    arestas = set()
    for link_group in links:
        for link in link_group:
            from_lane = link[0]
            edge = from_lane.rsplit("_", 1)[0]
            arestas.add(edge)
    return list(arestas)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)

    if not os.path.exists(CONFIG_FILE):
        print(f"ERRO: Arquivo '{CONFIG_FILE}' não encontrado!")
        sys.exit(1)

    sumo_cmd = [
        SUMO_BINARY,
        "-c", CONFIG_FILE,
        "--start",
        "--quit-on-end",
        "--no-warnings", "true",
    ]

    print("Iniciando simulação e coleta de dados para ML...")
    traci.start(sumo_cmd)

    todos_tls = traci.trafficlight.getIDList()
    if TLS_ID not in todos_tls:
        print(f"ERRO: Semáforo '{TLS_ID}' não encontrado!")
        traci.close()
        sys.exit(1)

    arestas_entrada = get_arestas_chegada(TLS_ID)

    dados_ml = []
    
    # Variáveis de controle de ciclo
    fase_atual_idx = traci.trafficlight.getPhase(TLS_ID)
    fase_atual_nome = traci.trafficlight.getPhaseName(TLS_ID)
    tempo_inicio_fase = 0.0

    # Acumuladores durante a fase ativa
    amostras_veiculos = []
    amostras_filas = []

    try:
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < TEMPO_MAXIMO:
            traci.simulationStep()
            tempo_atual = traci.simulation.getTime()

            # 1. Mede a demanda no instante atual
            veiculos_momento = sum(traci.edge.getLastStepVehicleNumber(e) for e in arestas_entrada)
            filas_momento = sum(traci.edge.getLastStepHaltingNumber(e) for e in arestas_entrada)

            amostras_veiculos.append(veiculos_momento)
            amostras_filas.append(filas_momento)

            nova_fase_idx = traci.trafficlight.getPhase(TLS_ID)

            # 2. Detecta a MUDANÇA DE FASE (transição entre GA e GS)
            if nova_fase_idx != fase_atual_idx:
                duracao_fase = tempo_atual - tempo_inicio_fase

                # Só registra se a fase teve um nome atribuído (ex: GA ou GS)
                if fase_atual_nome in ["GA", "GS"]:
                    # Médias e máximos da demanda durante o tempo em que o semáforo esteve nessa fase
                    media_veiculos = sum(amostras_veiculos) / len(amostras_veiculos) if amostras_veiculos else 0
                    max_veiculos = max(amostras_veiculos) if amostras_veiculos else 0
                    media_fila = sum(amostras_filas) / len(amostras_filas) if amostras_filas else 0
                    max_fila = max(amostras_filas) if amostras_filas else 0

                    dados_ml.append({
                        "tempo_inicio_s": round(tempo_inicio_fase, 1),
                        "tempo_fim_s": round(tempo_atual, 1),
                        "nome_fase": fase_atual_nome,              # GA ou GS
                        "fase_index": fase_atual_idx,              # 0 ou 1
                        "media_veiculos": round(media_veiculos, 2), # Média de carros na aproximação
                        "max_veiculos": max_veiculos,              # Pico de carros na aproximação
                        "media_fila": round(media_fila, 2),         # Média de carros parados
                        "max_fila": max_fila,                      # Pico da fila
                        "duracao_fase_s": round(duracao_fase, 1)    # TARGET / RÓTULO (Tempo aberto)
                    })

                # Reseta acumuladores para a nova fase
                fase_atual_idx = nova_fase_idx
                fase_atual_nome = traci.trafficlight.getPhaseName(TLS_ID)
                tempo_inicio_fase = tempo_atual
                amostras_veiculos = []
                amostras_filas = []

    except KeyboardInterrupt:
        print("\nSimulação interrompida.")
    finally:
        traci.close()

    # Salva o arquivo final
    df = pd.DataFrame(dados_ml)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Dataset para ML salvo em: {OUTPUT_CSV}")
    print(f"Total de fases registradas: {len(df)}")
    
    if not df.empty:
        print("\nExemplo das primeiras linhas do Dataset:")
        print(df.head(6).to_string())

if __name__ == "__main__":
    main()