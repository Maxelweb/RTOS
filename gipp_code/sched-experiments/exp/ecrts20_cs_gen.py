import random
import unionfind

MAX_RESOURCES = 32
MAX_GROUP_SIZE = 5

GROUP_TYPE_ALL = 0
GROUP_TYPE_WIDE = 1
GROUP_TYPE_DEEP = 2

# RTOS: necessary --> relazione di dipendenza tra risorse (padre, figlio) (?)

group_1 = dict()
group_1["num_res"] = 1
group_1["necessary"] = [(0,)]
group_1["top"] = [(0,)]
group_1["all"] = [(0,)]

group_2 = dict()
group_2["num_res"] = 2
group_2["necessary"] = [(0, 1)]
group_2["top"] = [(0,)]
group_2["all"] = [(1,)] + group_2["necessary"] + group_2["top"] #(1,),(0, 1),(0,)  0->1

group_3_wide = dict()
group_3_wide["num_res"] = 3
group_3_wide["necessary"] = [(0, 2), (1, 2)]
group_3_wide["top"] = [(0,), (1,)]
group_3_wide["all"] = [(2,)] + group_3_wide["necessary"] + group_3_wide["top"]

group_3_deep = dict()
group_3_deep["num_res"] = 3
group_3_deep["necessary"] = [(0, 1), (0, 2)]
group_3_deep["top"] = [(0,)]
group_3_deep["all"] = [(1,), (2,)] + group_3_wide["necessary"] + group_3_wide["top"]

group_4_wide = dict()
group_4_wide["num_res"] = 4
group_4_wide["necessary"] = [(0, 3), (1, 3), (2, 3)]
group_4_wide["top"] = [(0,), (1,), (2,)]
group_4_wide["all"] = [(3,)] + group_4_wide["necessary"] + group_4_wide["top"]

group_4_deep = dict()
group_4_deep["num_res"] = 4
group_4_deep["necessary"] = [(0, 2), (1, 2), (2, 3)]
group_4_deep["top"] = [(0,), (1,)]
group_4_deep["all"] = [(2,), (3,)] + group_4_deep["necessary"] + group_4_deep["top"]

# Non è possibile avere una CPU con un numero di core pari a 5, pertanto non possono esistere 5 istanze RNLP/Gruppi per 5 CPU
group_5_wide_1 = dict()
group_5_wide_1["num_res"] = 5
group_5_wide_1["necessary"] = [(0, 4), (1, 4), (2, 4), (3, 4)]
group_5_wide_1["top"] = [(0,), (1,), (2,), (3,)]
group_5_wide_1["all"] = [(4,)] + group_5_wide_1["necessary"] + group_5_wide_1["top"]

group_5_wide_2 = dict()
group_5_wide_2["num_res"] = 5
group_5_wide_2["necessary"] = [(0, 3), (1, 3), (1, 4), (2, 4)]
group_5_wide_2["top"] = [(0,), (1,), (2,)]
group_5_wide_2["all"] = [(3,), (4,)] + group_5_wide_2["necessary"] + group_5_wide_2["top"]

group_5_deep = dict()
group_5_deep["num_res"] = 5
group_5_deep["necessary"] = [(0, 3), (1, 3), (2, 3), (3, 4)]
group_5_deep["top"] = [(0,), (1,), (2,)]
group_5_deep["all"] = [(3,), (4,)] + group_5_deep["necessary"] + group_5_deep["top"]

def generate_group_cs_for_taskset(
    num_tasks,
    group_desc,
    num_requests_max,
    top_p = 0.5
):

    accesses = [[] for _ in xrange(0, num_tasks)]

    necc = len(group_desc["necessary"]) # necessary critical sections
    necc_count = 0
    if num_tasks < necc:
        print "not possible to generate accesses with %d tasks for a group with %d necessary critical sections" % \
        (num_tasks, necc)
        return []

    for i in xrange(0, num_tasks):

        if necc_count < necc:
            necc_access = True

        # num_requests = random.randint(1, num_requests_max)
        num_requests = num_requests_max[i] # RTOS: access amounts è un vettore [randint(1, acc_max_ls)] ripetuto n_ls volte --> vedi ecrts20.py ~riga 500

        for _ in xrange(0, num_requests):

            t_cs = None # tuple of critical section
            if necc_access: # Se ho meno necessary rispetto al totale
                t_cs = group_desc["necessary"][necc_count]
                necc_count += 1
                necc_access = False
            else:

                # Se probabilità soddisfatta prendo TOP
                # altrimenti prendo ALL (quindi anche possibili TOP)

                pick_top = random.random() <= top_p # RTOS: top probability da 0 a 1, se è minore stretto di 0.5
                pick_set = group_desc["top"] if pick_top else group_desc["all"] # Esegue la top probability e prende uno tra i top, se soddisfatta oppure ne prende uno qualunque
                # RTOS: WARNING!! POSSIBILE ERRORE: perché prende all e non soltanto necessary? ref. master thesis
                r = random.randint(0, len(pick_set)-1)
                t_cs = pick_set[r]

            accesses[i].append(list(t_cs)) # [j][k][l] --> j = num_task; k = lista (con solo un elemento?); l = tupla 

            # acc = list(list(tuple), list(...), ...)

    return accesses



# RTOS: Dalla definizione 6.0.1 si denota la presenza di un request symmetry ratio, cioè una misura di quanto simili sono le resource access patterns di due task. 
#       1) Viene innanzitutto individuato un valore di simmetria basato su due task, definito come il rapporto tra: 
#           - La sommatoria del numero di accessi effettuati da entrambi i task sulle risorse che si contendono (a cui accedono entrambi);
#           - Il numero di accessi totali che i due task effettuano
#       2) Individuato il valore si dice che i due task hanno:
#           - Alti pattern di accesso assimetrico (HAAPs) se symr(T_i, T_k) < x (un certo valore definito a priori);
#           - Altrimenti si dice che hanno pattern di accessi uniformi (UAPs) symr(T_i, T_k) >= x
#       3) Prendendo in esame due task T_i e T_k, individuando delle relazioni tra risorse l1 > l3, l2 > l3 e analizzando N_i,1 N_k,2 (Numero di richieste del task per una particolare risorsa) vogliamo:
#           - Un elevato valore per entrambi gli N in modo da ridurre il valore di simmetria fino ad un paramentro definito in fase di progettazione (x);
#           - In una situazione di HAAPs ci si aspetta che un fine-grained locking protocol dia un pi-blocking bound inferiore rispetto a un protocollo che realizza nested locking con group locks.
#           - Sotto GIPP l'unica richiesta che potrebbe bloccare J_k è quella per l2 effettuata da J_i. Il motivo è che, indipendentemente dal numero di replicas, solo il token top della coda di accesso può accedervi, accodando ogni altra eventuale richiesta.
def generate_asymmetric_group_cs_for_taskset(
    num_tasks,
    group_desc,
    num_requests_max,
    top_p = 0.9 # --> x (è la threshold?)
):

    accesses = [[] for _ in xrange(0, num_tasks)]

    necc = len(group_desc["necessary"])
    necc_count = 0
    if num_tasks < necc:
        print "not possible to generate accesses with %d tasks for a group with %d necessary critical sections" % \
        (num_tasks, necc)
        return []

    for i in xrange(0, num_tasks):

        if necc_count < necc:
            necc_access = True

        # num_requests = random.randint(1, num_requests_max)
        num_requests = num_requests_max[i] # --> numero massimo di richieste (generato randomicamente vedi ecrts20 vettore con valori rand)

        for _ in xrange(0, num_requests):

            t_cs = None
            if necc_access:
                t_cs = group_desc["necessary"][necc_count]
                necc_count += 1
                necc_access = False
            else:

                pick_top = random.random() <= top_p

                # prende solo elementi TOP
                if (pick_top): 
                    num_top = len(group_desc["top"])
                    pick_index = i % num_top # i-esimo task. Divide il numero del task per il numero di risorse top del gruppo e ne estrae il modulo
                    t_cs = group_desc["top"][pick_index]

                # prende solo elementi NON TOP
                else: 
                    pick_set = [x for x in group_desc["all"] if (x not in group_desc["top"])] # set with [necessary + non_top_res]
                    r = random.randint(0, len(pick_set)-1)
                    t_cs = pick_set[r]

            accesses[i].append(list(t_cs))

    return accesses


def assign_tasks_to_groups(num_tasks, min_tasks_per_group, num_groups):
    """
    For a given number of tasks, and a minimum number of tasks that need to access
    each group, return an array where the i-th element denotes which group the i-th
    task accesses. Each group is guaranteed to be accesses by at least min_tasks_per_group
    number of tasks.

    num_tasks - int:
        number of tasks to assign to groups

    min_tasks_per_group - int:
        minimum number of tasks assignment to each group

    num_groups - int:
        numer of groups to assign tasks to
    """

    # [-1 -1 -1 -1 ... -1_numtasks]
    group_assignments = [-1 for _ in xrange(0, num_tasks)]

    # Itero su tutti i gruppi
    for i in xrange(0, num_groups):

        group_task_count = 0

        while True:
            
            # se r = 3, allora [-1 -1 -1 i -1 ...]
            r = random.randint(0, num_tasks-1)
            if group_assignments[r] == -1:
                group_assignments[r] = i
                group_task_count += 1

            if group_task_count == min_tasks_per_group:
                break
    
    # POST: ho un numero minimo di task assegnati a ciascun gruppo


    for i in xrange(0, num_tasks):

        if group_assignments[i] != -1:
            continue

        # Se un task non è stato ancora assegnato (-1), assegno un gruppo randomico (r) 
        r = random.randint(0, num_groups-1)
        group_assignments[i] = r

    # POST: tutti i task sono assegnati a un gruppo
    
    return group_assignments
