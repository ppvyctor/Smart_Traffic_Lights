"""
Coleta de Dados baseada em Estados e Recompensas em Tempo Real para RL.
Gera um CSV formatado com observações contínuas por intervalo de decisão.
"""

import os
import sys
import traci
import pandas as pd

# ==================== CONFIGURAÇÕES ====================
SUMO_BINARY = "sumo"              # Use "sumo-gui" para visualizar na interface
CONFIG_FILE = "teste.sumocfg"     # Arquivo de configuração do SUMO
OUTPUT_CSV = "dataset_ml_estado_recompensa.csv"

TEMPO_MAXIMO = 10800               # 1 hora de simulação (3600 segundos)
TLS_ID = "J2"                     # ID do semáforo
INTERVALO_DECISAO = 5             # Intervalo de amostragem/decisão em segundos
# =======================================================

# Mapeamento das vias do seu cruzamento no SUMO:
# Leste-Oeste: E0 e -E1
# Norte-Sul: -E3 e -E2
VIAS_LESTE_OESTE = ["E0", "-E1"]
VIAS_NORTE_SUL = ["-E3", "-E2"]

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

    print("Iniciando simulação e coleta de estados/recompensas...")
    traci.start(sumo_cmd)

    todos_tls = traci.trafficlight.getIDList()
    if TLS_ID not in todos_tls:
        print(f"ERRO: Semáforo '{TLS_ID}' não encontrado!")
        traci.close()
        sys.exit(1)

    dados_ml = []
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < TEMPO_MAXIMO:
            # Avança a simulação em passos curtos até o próximo intervalo de decisão
            tempo_atual = traci.simulation.getTime()
            proximo_passo = tempo_atual + INTERVALO_DECISAO

            while tempo_atual < proximo_passo and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                tempo_atual = traci.simulation.getTime()

            # --- PASSO 1: Leitura do Estado (State/Observation) ---
            fila_lo = sum(traci.edge.getLastStepHaltingNumber(e) for e in VIAS_LESTE_OESTE)
            fila_ns = sum(traci.edge.getLastStepHaltingNumber(e) for e in VIAS_NORTE_SUL)
            
            espera_lo = sum(traci.edge.getWaitingTime(e) for e in VIAS_LESTE_OESTE)
            espera_ns = sum(traci.edge.getWaitingTime(e) for e in VIAS_NORTE_SUL)

            fase_index = traci.trafficlight.getPhase(TLS_ID)
            fase_nome = traci.trafficlight.getPhaseName(TLS_ID)

            # --- PASSO 2: Cálculo da Recompensa (Reward) ---
            # Quanto maior o tempo total de espera e as filas, menor é a recompensa (penalização)
            espera_total = espera_lo + espera_ns
            fila_total = fila_lo + fila_ns
            recompensa = - (espera_total + (fila_total * 0.5))

            dados_ml.append({
                "tempo_s": round(tempo_atual, 1),
                "fase_index": fase_index,
                "fase_nome": fase_nome,
                "fila_leste_oeste": fila_lo,
                "fila_norte_sul": fila_ns,
                "espera_leste_oeste_s": round(espera_lo, 1),
                "espera_norte_sul_s": round(espera_ns, 1),
                "fila_total": fila_total,
                "espera_total_s": round(espera_total, 1),
                "recompensa": round(recompensa, 2)
            })

    except KeyboardInterrupt:
        print("\nSimulação interrompida manualmente.")
    finally:
        traci.close()

    df = pd.DataFrame(dados_ml)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Dataset de Estados e Recompensas salvo em: {OUTPUT_CSV}")
    print(f"Total de passos registrados: {len(df)}")
    if not df.empty:
        print("\nPrimeiras linhas registradas:")
        print(df.head(6).to_string())

if __name__ == "__main__":
    main()