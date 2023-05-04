# RedesNeurais_CesarSchool

Tales Araújo Carvalho

Atividade: Dataset / Detecção de objetos / YOLO v5 PyTorch

Geral: dataset composto por skins do jogo eletrônico Counter-Strike Global Offensive, com 300 imagens distribuídas em 4 classes (75 por classe).

Skins são itens utilizados dentro do jogo as quais se diferem pela sua forma física (cor, nível de desgaste, brilho). Cada skin é exclusiva e única! Elas são comercializadas pela comunidade e seus preços variam de acordo com a raridade além das outras variáveis citadas anteriormente.

Para o dataset foram utilizados quatro classes de skins: AK-47, USP, AWP e MP9.
Cada classe de skin possui diversos modelos distintos dentro da própria classe. Por exemplo, a AK-47 tem uma skin chamada "Asiimov" de cor branca e outra skin chamada de "Redline" a qual possui cor preta e linhas vermelhas. É possível encontrar mais de uma skin "Asiimov" no inventário de um mesmo player (o que varia de uma pra outra são as condições da skin: nível de desgaste, etc). O mesmo pensamento se aplica para todas as armas do jogo, logo funciona também para as outras classes utilizadas no dataset. É normal então vermos skins do mesmo modelo no dataset.

