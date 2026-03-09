
"""BPE refactor skeleton for Tokenized_numpymatrixversion (Master/Worker)."""
#TOKENIZER using bpe method

import os
from tqdm import tqdm
from collections import Counter, defaultdict
import multiprocessing as mp
import numpy as np
from typing import Tuple, Dict, Any, List
from multiprocessing.connection import Connection
from array import array
import resource


# Définit la limite à 80 Go (100 * 1024 * 1024 * 1024 octets)
limit = 80 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

#debug config
DELETED_MARKER = 65535

# Config
TEXT_PATH = "/home/gaspard/projets/part1_combined_training_text.bin"
#TEXT_PATH = "/home/gaspard/projets/combined_training_text.bin"
#!/usr/bin/python3
TOKEN_NUMBER = 2000



class BPEMaster:
    """Master controller pour l'entraînement BPE (Stateful Workers + Deltas)."""
    

    def __init__(self, worker_count: int = 4):
        self.worker_count = worker_count #int : récupère l'argument passé
        self.workers = []  #List[mp.Process]
        self.conns = [] #List[mp.connection.Connection] (master ends of pipes)
        
        self.global_counter = Counter() #Counter mapping pair->count 
        
        #nd : les valeurs 0 à 255 sont réservées aux bytes bruts (uint8)
        self.next_token_id = 256 #int (pr next id)
        
        self.vocab = {} #Dict[repr, id] : règles de fusion ({256: (101, 101)})
        self.worker_chunks_meta = [] #List[Dict] : metadata sur les chunks (offsets, lengths)
        

    def init_workers(self, file_path: str, config: Dict[str, Any] = None) -> None:
        """Créer et démarrer les workers stateful en leur donnant des offsets de lecture.
        Inputs:
        - file_path:  (.bin)
        - config: options diverses
        """

        #Calcul découpe
        file_size = os.path.getsize(file_path)
        chunk_size = file_size // self.worker_count
        
        print(f"[Master] Initialisation de {self.worker_count} workers sur {file_path} ({file_size/1024/1024/1024:.2f} Go)")

        #Boucle de lancement
        for i in range(self.worker_count):
            #Calcul des bornes pour ce worker
            start_offset = i * chunk_size
            length = chunk_size if i < self.worker_count - 1 else file_size - start_offset # nd : last worker : prend le reste
        
            #self.worker_chunks_meta.append({'id': i, 'offset': start_offset, 'length': length})    # metadonnees pr le debug
            
            # Création du Pipe 
            master_conn, worker_conn = mp.Pipe(duplex=True)
            
            # Création du Process

            # PAS de numpy array ici, juste le chemin+chiffres
            p = mp.Process(
                target=bpe_worker_process, 
                args=(i, worker_conn, file_path, start_offset, length, config)
            )
            
            p.start() #Démarrage
            
            worker_conn.close()  #cleanup
            
            # Enregistrement
            self.workers.append(p)
            self.conns.append(master_conn)
            
        print("[Master] workers démarrés. attente des compteurs...")


    def collect_initial_counters(self, timeout: float = None) -> None:
        """Collecte et fusionne les counters initiaux envoyés par chaque worker."""
        print(f"[Master] Reception des stats de {self.worker_count} workers...")
        
        for i, conn in enumerate(tqdm(self.conns, desc="Fusion des compteurs")):
            # Timeout Failsafe
            if conn.poll(timeout): 
                try:
                    msg = conn.recv() #thx gemmini
                except EOFError:
                    print(f"[Erreur] Le worker {i} a fermé la connexion inopinément.")
                    continue
            
                if not isinstance(msg, tuple) or len(msg) != 2:
                    print(f"[Erreur] Message invalide reçu du worker {i}: {msg}")
                    continue
                
                tag, local_counter = msg
                
                if tag == 'INIT' and isinstance(local_counter, Counter):    # ----> cas good normal
                    self.global_counter.update(local_counter)# fusion dans le Global Counter
                else:
                    print(f"[Erreur] Protocole non respecté par le worker {i}. Reçu: {tag}") #fs
            else:
                print(f"[Timeout] Le worker {i} ne répond pas (délai: {timeout}s).")#fs
        

        # Nd calcul : Commence apres 255, si on trouve un token ID 500 dans le texte,
        # le prochain token créé devra être 501 (pas 256)

        max_existing_id = 255
        if self.global_counter:
            # GORSSE INIT
            for pair in self.global_counter.keys():
                max_existing_id = max(max_existing_id, pair[0], pair[1])
        
        self.next_token_id = max_existing_id + 1
        
        print(f"[Master] Initialisation terminée.")
        print(f"   - Paires uniques : {len(self.global_counter)}")
        print(f"   - Prochain Token ID : {self.next_token_id}")
    

    def training_loop(self, merges: int):
        """Boucle principale pour effectuer `merges` fusions."""
        
        print(f"[Master] Démarrage de l'entraînement pour {merges} itérations...")
        pbar = tqdm(range(merges), desc="BPE Training")
        
        for i in pbar:
            # Sélectionner la paire la plus fréquente
            
            if not self.global_counter:
                print("[Info] Plus aucune paire répétée trouvée. Arrêt prématuré.") #fs débuggage
                break
            

            best_pair, freq = self.global_counter.most_common(1)[0] # most_common(1) -> tuple (a,b) freq -> int

            new_id = self.next_token_id
            self.next_token_id += 1
            
            # ajout au dico vocabulaire
            # Format : { 256 : (101, 120) } -> Le token 256 est fait de 101 et 120
            self.vocab[new_id] = best_pair
            
            pbar.set_postfix({'Pair': str(best_pair), 'NewID': new_id, 'Freq': freq})
            

            self.send_merge_to_workers(best_pair, new_id) #envoi de l'ordre aux workers
            

            # --> Recevoir les Deltas (Attente active) (function blocante)
            #nd reçois des +/-
            aggregated_deltas = self.receive_deltas()
            self.apply_delta(aggregated_deltas) #applique les delta au glocal counter
            
            
        print("[Master] Boucle terminée.")


        # Rapatriement + Sauvegarde
        print("[Master] Arrêt des workers et récupération du texte final...")
        final_chunks = self.stop_workers()
        
        # Sauvegarde du texte binaire compressé (uint16)
        output_bin = "text_tokenized.bin"
        print(f"[Master] Écriture de {output_bin}...")
        with open(output_bin, 'wb') as f:
            for chunk in final_chunks:
                chunk.tofile(f)
                
        # Sauvegarde du vocabulaire (JSON)
        output_vocab = "vocab.json"
        print(f"[Master] Écriture de {output_vocab}...")
        import json
        
        # nd : Conversion pour JSON : Les clés doivent être des strings, et les tuples des listes
        # { "256": [101, 120] }
        serializable_vocab = {
            str(k): [int(v[0]), int(v[1])] for k, v in self.vocab.items()
        }
        
        with open(output_vocab, 'w', encoding='utf-8') as f:
            json.dump(serializable_vocab, f, indent=4)
            
        print("[Master] Terminé")

    def send_merge_to_workers(self, pair: Tuple[int, int], new_id: int) -> None:
        """('MERGE', pair, new_id) à tous les workers.
        Inputs:
        - pair: tuple (A, B)
        - new_id: id du token fusionné"""


        #TODO 
        #gérer erreurs d'envoi et reconnexion éventuelle (auto reco)

        payload = ('MERGE', pair, new_id)
        
        for conn in self.conns :
            try :
                conn.send(payload)
            except (BrokenPipeError, EOFError, ConnectionResetError) as e: #fs
                print(f"erreur de communication avec le worker code : {e}")

    def receive_deltas(self, timeout: float = None) -> Counter:
        """Recevoir un message ('DELTA', delta_dict) de chaque worker et agréger.
        Outputs:
        - aggregated Counter (pair -> delta_count) nd : peux négatifs"""

        aggregated = Counter()

        for conn in self.conns:
            message = conn.recv() #voir pr ajouter une failsafe timeout
            if isinstance(message, tuple) and len(message) == 2 and message[0] == 'DELTA': #check nb d'éléments + si c'est un delta
                aggregated.update (message[1])
        
        return aggregated



    def apply_delta(self, delta: Counter) -> None:
        """Appliquer le delta agrégé à self.global_counter.
        - mettre à jour les comptes, supprimer paires avec compte <= 0"""
        
        for pair, change in delta.items():
             new = self.global_counter.get(pair, 0) + change

             if new > 0: self.global_counter[pair] = new
             else: #supp pour opti
                self.global_counter.pop(pair, None)


    def stop_workers(self, timeout: float = None) -> List[np.ndarray]:
        """ordonne ('STOP',) à tous les workers + collection ('FINAL', np.array).
        Return
        - list de numpy arrays (ds ordre des workers)"""
        
        for conn in self.conns :#envoi de l'ordre
            conn.send(('STOP',))
        
        final_arrays = []

        for conn in self.conns:
            final_array = conn.recv() #voir pr ajouter une failsafe timeout
            if isinstance(final_array, tuple) and len(final_array) == 2 and final_array[0] == 'FINAL': #check nb d'éléments + si c'est un FINAL
                final_arrays.append(final_array[1])
            else : print("erreur de récéption des résultats (worker stop)")
            

        return final_arrays


def bpe_worker_process(worker_id: int, conn: Connection, file_path : str, start_offset : int, length : int, config: Dict[str, Any] = None):
    """Worker stateful """

    # TODO: Initialisation
    # - Définir un marqueur pour les noeuds supprimés (p.ex. values[i] = -1) si utilisé
    # - Calculer le Counter initial des paires locales et envoyer:
    #       conn.send(('INIT', counter_dict))

    #init
    initial_chunk = np.fromfile(file_path, dtype=np.uint8, count=length, offset=start_offset)
    size_temp = initial_chunk.size
    values = initial_chunk.astype(np.uint16, copy=True) #en cas de scale mettre uint32
    del initial_chunk 
    left = np.arange(0, size_temp, dtype=np.uint32) #voir pour ptet mettre uint 16/64
    right = np.arange(2, size_temp + 2, dtype=np.uint32)
    right[-1] = 0 #nd : je fais un décalage de valeurs : 0 = -1

    del size_temp

    # Création de l'index des positions (array C ultra léger)
    positions = defaultdict(lambda: array('I'))
    
    for i, (a, b) in enumerate(zip(values[:-1], values[1:])):
        positions[(a, b)].append(i)

    # Initialisation du compteur à partir de la taille des listes
    initial_pairs = Counter({pair: len(arr) for pair, arr in positions.items()})
    conn.send(('INIT', initial_pairs))

    
    
    

    while True:
         message = conn.recv()
         if message[0] == 'MERGE':
             pair, new_id = message[1], message[2]
             delta = Counter()
             # Récupération directe O(1)
             if pair not in positions:
                 conn.send(('DELTA', dict()))
                 continue
             candidates = positions[pair]

             #application fusions
             for i in candidates:
                 if values[i] != pair[0]: continue #check anti chevauchement
                 r_ptr = right[i]
            
                 if r_ptr == 0: continue #recup sécurisé
                 j = r_ptr - 1

                 if values[j] != pair[1]: continue
                    
                 l_ptr = left[i]     # Voisin à gauche de A
                 rr_ptr = right[j]   # Voisin à droite de B

                #supp ancienens paires
                 delta[pair] -= 1
                 if l_ptr != 0:
                    left_val = values[l_ptr - 1]
                    delta[(left_val, pair[0])] -= 1
                 if rr_ptr != 0:
                    right_val = values[rr_ptr - 1]
                    delta[(pair[1], right_val)] -= 1
                
                #MAJ pointeurs
                 values[i] = new_id
                 values[j] = DELETED_MARKER
                 right[i] = rr_ptr
                 if rr_ptr != 0: left[rr_ptr - 1] = i + 1

                 #ajout new pairs de la fusion
                 if l_ptr != 0:delta[(left_val, new_id)] += 1
                 if rr_ptr != 0:delta[(new_id, right_val)] += 1

                 # MAJ de l'index avec les nouvelles paires
                 if l_ptr != 0:
                     positions[(left_val, new_id)].append(l_ptr - 1)
                 if rr_ptr != 0:
                     positions[(new_id, right_val)].append(i)

                #cela créé des index fantomes qui bouffe de la ram plus le nombre d'itération est grande --> risque de crash
                #pour regler ça : TODO garbage collector

                

             del positions[pair]
             #return
             conn.send(('DELTA', dict(delta)))
                    


         elif message[0] == 'STOP':
             #comptage final
             final_array = values[values != DELETED_MARKER]
             conn.send(('FINAL', final_array))
             break

    


__all__ = ['BPEMaster', 'bpe_worker_process']

if __name__ == '__main__':
    workers = mp.cpu_count() - 1 
    if workers < 1: workers = 1
    
    print(f"--- Démarrage du Tokenizer BPE avec {workers} processus ---")

    master = BPEMaster(worker_count=workers)
    
    master.init_workers(TEXT_PATH)
    master.collect_initial_counters(timeout=60.0) # On laisse 60 secondes max pour lire le SSD
    
    master.training_loop(merges=TOKEN_NUMBER)