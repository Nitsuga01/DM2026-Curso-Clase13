---
title: No13 - Programación Diferencial Pt3
---

# Programación Diferenciable: Métodos Reverse Pt1

**Fecha:** 03/06/2026

:::{iframe} https://www.youtube.com/embed/bBnnk7YehsE
:width: 100%
:::

## Programación diferenciable utilizando métodos reverse

### Repaso de la clase anteirior: Diferenciación automática en modo forward

En la [clase 12](clase12.md), se estudió el método de *Automatic Differentiation (AD)* en el caso *Forward*, implementado mediante los números duales. Repasaremos a continuación los conceptos de este caso que reaparecerán en esta clase.
Para realizar la diferenciación, consideramos la cantidad a diferenciar como el nodo final de un *grafo computacional* cuyas entradas son los $p$ parámetros de nuestro problema.
```{figure} ../images/grafocomp.png
:width: 300px
:align: center
:name: grafo-computacional

Epígrafe de la figura. Se puede referenciar como {numref}`fig-ejemplo`.
```

A partir de la fórmula de Bauer, la derivada de la salida respecto a algún parámetro es la suma de los productos de derivadas correspondientes a los diferentes caminos del gráfo que los unen (ver [clase 11](clase11.md)).
$$\frac{\partial v_m}{\partial v_o}= \sum_{v_o \underset{\Gamma}{\to} v_m} \prod_{w_k\to w_{k+1}\in \Gamma} \frac{\partial w_{k+1}}{\partial w_k}.$$

### Costo computacional en la fórmula de Bauer, métodos de evaluación.
La eficiencia en la evaluación de esta fórmula depende del método utilizado. 
Por ejemplo, el método *naíf* de evaluación, listar todos los caminos, calcular el producto para cada camino, y luego efectuar la suma, es altamente ineficiente. Esto es porque muchos caminos compartirán secciones, que serán calculadas multiples veces sin reusar resultados. Por ejemplo, en la figura se muestran todas las aristas relevantes para calcular $\partial v_{10}/\partial v_0$. No es difícil notar que habrá más que un camino que une los nodos relevantes y que una buena parte de ellos utilizará la arista que une $v_0$ y $v_1$ y por lo tanto la derivada parcial $\partial v_1/\partial v_0$.
```{figure} ../images/grafocompderivada.png
:width: 300px
:align: center
:name: grafo-computacional-derivada

Epígrafe de la figura. Se puede referenciar como {numref}`fig-ejemplo`.
```

### El método Forward
Otro método, empleado la clase pasada, consiste en explorar el grafo en un orden tal que cada nodo sea explorado únicamente cuando se visitaron todos los nodos previos de los caminos de los que forma parte (ver ejemplo en figura). Esto permite unificar aquellos caminos que comparten secciones y reducir el costo computacional. Al ir acumulando los cálculos parciales de la fórmula de Bauer, se irán calculando las derivadas parciales de nodos de índice cada vez más altos. Así, podríamos decir que la estrategia _Forward_ para encontrar la derivada $\partial v_j/\partial v_i$ con $j>i$ consiste en encontrar cantidades $\omega$ previas a $v_j$ y reescribir el problema de acuerdo a la siguiente relación dinámica:
$$\frac{\partial v_j}{\partial v_i} = \sum_{\omega \to v_j} \frac{\partial v_j}{\partial \omega}\frac{\partial \omega}{\partial v_i}.$$
Al ser $\omega$ inmediatamente próximo a $v_j$, la derivada $\frac{\partial v_j}{\partial \omega}$ será sencilla de calcular, restando la dificultad en calcular $\frac{\partial \omega}{\partial v_i}$, lo que se hará de forma recursiva reaplicando la fórmula.

En nuestro ejemplo, comenzaremos calculando $\frac{\partial v_0}{\partial v_0}$, luego $\frac{\partial v_1}{\partial v_0}$ y $\frac{\partial v_2}{\partial v_0}$, luego $\frac{\partial v_4}{\partial v_0}$, $\frac{\partial v_5}{\partial v_0}$ y $\frac{\partial v_6}{\partial v_0}$, luego $\frac{\partial v_8}{\partial v_0}$ y $\frac{\partial v_9}{\partial v_0}$ y por último el resultado buscado $\frac{\partial v_{10}}{\partial v_0}$. Ya que estamos calculando las derivadas de nodos cada vez más cercanos a  la salida siguiendo el ordenamiento del grafo, llamamos a este el método _Forward_.
```{figure} ../images/grafocompforward.png
:width: 300px
:align: center
:name: grafo-computacional-derivada

Epígrafe de la figura. Se puede referenciar como {numref}`fig-ejemplo`.
```

## El método Reverse

Análogamente al método forward que consistía en explorar los nodos del grafo de forma ordenada partiendo de los parámetros y llegando al valor de salida, el método *Reverse* consistirá en explorar los nodos del grafo en un orden similar pero partiendo del valor de salida y dirigiéndose hacia los parámetros, calculando en el proceso derivadas de la salida respecto a valores de cada vez menor índice. Así, el método reverse para calcular una derivada $\frac{\partial v_j}{\partial v_i}$ con $j>i$ también hará uso de cantidades intermedias $\omega$ que sucedan al nodo $v_i$ y se valdrá de la relación dinámica:
$$\frac{\partial v_j}{\partial v_i} = \sum_{v_i\to \omega} \frac{\partial v_j}{\partial \omega}\frac{\partial \omega}{\partial v_i}.$$
De las dos derivadas a la derecha, esta vez será $\frac{\partial \omega}{\partial v_i}$ la que es fácil de calcular mientras que para $\frac{\partial v_j}{\partial \omega}$ se tendrá que aplicar la relación recursivamente. A este método también se le da el nombre *back-propagation*.

Reusando el ejemplo anterior, se calcularán primero $\frac{\partial v_{10}}{\partial v_8}$ y $\frac{\partial v_{10}}{\partial v_9}$, luego $\frac{\partial v_{10}}{\partial v_4}$, $\frac{\partial v_{10}}{\partial v_5}$ y $\frac{\partial v_{10}}{\partial v_6}$, luego $\frac{\partial v_{10}}{\partial v_1}$ y $\frac{\partial v_{10}}{\partial v_2}$ y por último $\frac{\partial v_{10}}{\partial v_0}$.
```{figure} ../images/grafocompbackwards.png
:width: 300px
:align: center
:name: grafo-computacional-derivada

Epígrafe de la figura. Se puede referenciar como {numref}`fig-ejemplo`.
```

### Uso de memoria y _checkpointing_


Una desventaja del modo backwards respecto al modo forwards, es que requiere precomputar los valores asignados a cada nodo para poder evaluar las derivadas. Por ejemplo, para calcular un valor para $\frac{\partial v_{j+1}}{\partial v_{j}}$, uno necesita del valor de $v_j$ o $v_{j+1}$ en que evaluar la derivada. En el modo forward, la obtención de estos valores a partir de los parámetros $v_{-p}, ... ,v_0$ o las variables secundarias que se deducen de estos puede hacerse en el momento. En el caso backwards, todos estos valores deberán conocerse al inicio del computo de la derivada. Esto acarrea un costo adicional en memoria para guardar todas las variables intermedias del programa y un costo en tiempo por el acceso de la memoria.
```{figure} ../images/casobackward.png
:width: 300px
:align: center
:name: grafo-computacional-derivada

Epígrafe de la figura. Se puede referenciar como {numref}`fig-ejemplo`.
```
```{figure} ../images/casoforward.png
:width: 300px
:align: center
:name: grafo-computacional-derivada

Epígrafe de la figura. Se puede referenciar como {numref}`fig-ejemplo`.
```

Para aliviar esta demanda de memoria, se suele usar la estrategia de *Checkpointing*. Esta estrategia consiste en guardar en memoria el estado del sistema únicamente en ciertos puntos espaciados del programa. Si luego para el cálculo de derivadas mediante algún metodo backwards se requieren valores intermedios a los puntos disponibles, estos deberán poder obtenerse corriendo el programa de vuelta a partir del punto guardado más cercano.
```{figure} ../images/checkpointing.png
:width: 300px
:align: center
:name: grafo-computacional-derivada

Epígrafe de la figura. Se puede referenciar como {numref}`fig-ejemplo`.
```

### Cuando conviene backwards? VJP vs JVP

Vale la pena pensar en que casos las ventajas de los métodos backwards los vuelven convenientes aún considerando las penalidades en memoria y computacional que estos implican. Para ver un ejemplo de estas ventajas, podemos considerar una versión vectorizada de un grafo computacional:
```{figure} ../images/grafovectorizado.png
:width: 300px
:align: center
:name: grafo-computacional-derivada

Epígrafe de la figura. Se puede referenciar como {numref}`fig-ejemplo`.
```
Aquí, los $p$ parámetros $\theta$ son la entrada del grafo computacional, mientras que la salida es la función de costo $\mathcal L$, un escalar real. En las capas intermedias, las variables intermedias $h_j\in \mathbb R^{d_j}$ se calculan como $h_j=g_j(h_{j-1})$, donde $g_j:\mathbb R^{d_{j-1}}\to \mathbb R^{d_j}$ es la función que relaciona las variables en cada nodo con el anterior. Así, la función de costo será $\mathcal L(\theta)=g_m\circ g_{m-1}\circ...\circ g_1(\theta)$.
Empleando la regla de la cadena, el Jacobiano de la función de costo será el producto de los Jacobianos de cada $g_m$:
$$D\mathcal L=Dg_m\;Dg_{m-1}\;...\;Dg_1 (\theta)$$
El tamaño de $D\mathcal L$ será $1\times p$, y cada $Dg_{j}$ tendrá tamaño $d_m \times d_{m-1}$. Podemos observar que en este caso el calculo del lado derecho de la ecuación por un método backwards (equivalente a calcular de izquierda a derecha el producto de matrices) será eficiente en comparación a los métodos forward (equivalentes a calcular de derecha a izquierda el producto de matrices). Como en este caso $d_m=1$, al hacer el producto de izquierda a derecha, se estará realizando la multiplicación de un vector por una matriz repetidas veces (Vector-Jacobian Product (VJP)). En cambio, como $d_0=p$, al realizar la multiplicación en sentido inverso (Jacobian-Vector Product (JVP)), se estarán multiplicando matrices por matrices hasta el final.

Para calcular $AB$ con $A\in \mathbb R^{n\times r}$ y $B\in \mathbb R^{r\times m}$ se requieren $n\times m$ productos internos, cada uno de $r$ términos, el costo de calcular $AB$ será de orden $\mathcal O(mnr)$. En general, en un caso en que tenemos una función de costo $\mathcal L:\mathbb R^{p}\to \mathbb R^q$ con $m$ pasos intermedios, el número de operaciones a realizar en modo forward va como $\mathcal O(pm+q)$, mientras que en el modo backward irá como $\mathcal O(mq+p)$.

Esto, junto al mayor costo en memoria de aplicar metodos backwards, sugieren que esto es únicamente conveniente si $p>q$.

|       | Forward | Reverse |
| ----- | ------- | ------- |
| $p<q$ | ✓       | X       |
| $p=q$ | ✓       | X       |
| $p>q$ | X       | ✓       |
Para dar números más concretos. Dada una ODE de $n$ variables, con $q=1$, para $n+p\gtrsim 50$, convene utilizar métodos reverse, mientras que en el caso contrario conviene utilizar métodos forward.


TODO: Rellenar los 15' que vimos del método del adjunto.
