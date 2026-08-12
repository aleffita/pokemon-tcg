# Capítulo 6: Memória Oculta, TBPTT e A Anomalia do Mean Dinâmico

A passagem de dados através das Camadas de Atenção ($L=4$) gera computações esparsas locais que colapsariam em entropia de esquecimento rápido caso a arquitetura fosse isolada. O Pokémon TCG exige planejamento sequencial.
Para sanar a miopia analítica e permitir raciocínios com duração temporal, o núcleo da *Policy* MLX reserva uma porção autônoma no espaço latente.

## 6.1. A Matriz Estocástica Persistente (Scratch Registers)
Embebidos fisicamente na arquitetura, existem **32 Scratch Tokens** (Registradores de Rascunho) com forma $(32, 128)$.
Diferente das Cartas da Mão ou do Oponente que entram e saem do *Stream* com o estado do turno, esses tokens não possuem amarras com a realidade física do jogo. Eles atuam exclusivamente como blocos de notas lógicos para a própria rede.
Durante uma iteração de *Flash Attention*, a matriz funde:
$$ \mathcal{H}_{in} = [ \text{State\_Tokens} \parallel \text{Scratch\_Tokens} \parallel \text{Option\_Tokens} ] $$

## 6.2. Fluxo Recorrente Temporal (TBPTT)
Se os registros reiniciassem a cada turno, sua serventia seria nula. O `bc_train_mlx.py` injeta a continuidade:
A cada passo (*step*) de um Episódio processado, o sistema extrai o *output* do tensor de *Scratch* que passou pela última Camada 4 e injeta fisicamente esses dados de volta como o `memory_in` do *Step* de decisão imediatamente posterior. 
Isso unifica a rede como uma Topologia Recorrente Atencional (TBPTT): a rede pode escrever uma conjectura no Registro no turno 1 (ex: "tenho os componentes, prepararei o boss no turno 3"), reter esse vetor através de múltiplas ações triviais (jogar *Nest Ball*, evoluir) e recuperar o vetor letal no momento da eclosão tática.

## 6.3. O Fator de Isolamento (Shock Absorber)
No Estágio 3 do Currículo, o motor sofreu um abalo estrutural. 
As 4 Cabeças Auxiliares (`ko_head_aux`, `prize_head_aux`, `terminal_head_aux` e `return_head_aux`) guiam a atenção em sub-módulos. Devido a uma normalização espacial deficitária no micro-loteamento espacial MLX (bug de *Mean* Dinâmico), apenas as linhas ativas eram agregadas, e a derivada BCE (*Binary Cross Entropy*) do `ko_head_aux` explodiu multiplicando o gradiente global num pico massivo de $\text{Loss} = 81.0$.

Em arquiteturas convencionais, gradientes escalados por fatores de $1000\times$ estilhaçam os pesos para $NaN$, obrigando um reinício do *Checkpoint*.
Contudo, o motor instanciado ignorou o colapso estelar. A válvula de escape (*Attention Sink*) funcionou perfeitamente: **os 32 Scratch Registers absorveram a bomba de gradiente**. 

O choque forçou os Registros a hipertrofiarem suas valências ao redor da heurística mais alta (O Nocaute — *KO*). Como subproduto matemático não-intencional, mas espetacular, a rede transferiu todo esse desespero letal para a sua tática orgânica, retendo no Estágio 3 uma letalidade letal imprevista que quebrou as barreiras de pontuação do *Behavioral Cloning*.
