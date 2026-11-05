from auto_hyperontology.core import Hyperontology, HyperNode, HyperEdge, BoundaryNode

def build_khan_academy_hyperontology() -> Hyperontology:
    h = Hyperontology(name="KhanAcademy_Curriculum_Hyperontology")

    math_nodes = [
        HyperNode(
            id="linear_equations",
            name="Linear Equations and Slopes",
            domains=["Mathematics"],
            categories=["Concept", "Process"],
            aliases=["Ecuaciones Lineales", "Pendiente"]
        ),
        HyperNode(
            id="polynomial_functions",
            name="Polynomial and Power Functions",
            domains=["Mathematics"],
            categories=["Concept"],
            aliases=["Funciones Polinomiales"]
        ),
        HyperNode(
            id="limits_and_continuity",
            name="Limits and Continuity",
            domains=["Mathematics"],
            categories=["Concept", "Property"],
            aliases=["Límites y Continuidad"]
        ),
        HyperNode(
            id="definite_integral",
            name="Definite Integral & Accumulation",
            domains=["Mathematics", "Physics", "Economics"],
            categories=["Concept", "Process"],
            aliases=["Integral Definida", "Área bajo la curva"]
        ),
        HyperNode(
            id="vectors_and_spaces",
            name="Vectors and Vector Spaces",
            domains=["Mathematics", "Physics", "ComputerScience"],
            categories=["Concept", "Property"],
            aliases=["Vectores", "Espacios Vectoriales"]
        ),
        HyperNode(
            id="matrix_transformations",
            name="Matrix Transformations & Tensors",
            domains=["Mathematics", "ComputerScience", "Physics"],
            categories=["Concept", "Process"],
            aliases=["Transformaciones Matriciales"]
        )
    ]

    derivative_node = BoundaryNode(
        id="derivative_rate_of_change",
        name="Derivative & Instantaneous Rate of Change",
        domains=["Mathematics", "Physics", "Economics", "ComputerScience"],
        categories=["Concept", "Process", "Property"],
        aliases=["Derivada", "Tasa de Cambio Instantánea", "Gradient"],
        domain_mappings={
            "Mathematics": "math.calculus.derivative",
            "Physics": "physics.kinematics.velocity",
            "Economics": "econ.micro.marginal_analysis",
            "ComputerScience": "cs.ai.gradient_descent"
        },
        bridge_degree=1.0
    )

    for n in math_nodes:
        h.add_node(n)
    h.add_node(derivative_node)

    physics_nodes = [
        HyperNode(
            id="kinematic_equations",
            name="Kinematics & 1D Motion",
            domains=["Physics"],
            categories=["Concept", "Process"],
            aliases=["Cinemática", "Movimiento Unidimensional"]
        ),
        HyperNode(
            id="instantaneous_velocity",
            name="Instantaneous Velocity & Acceleration",
            domains=["Physics"],
            categories=["Property", "Process"],
            aliases=["Velocidad Instantánea", "Aceleración"]
        ),
        HyperNode(
            id="work_energy_theorem",
            name="Work-Energy Theorem",
            domains=["Physics"],
            categories=["Concept", "Level"],
            aliases=["Teorema del Trabajo y la Energía"]
        )
    ]
    for n in physics_nodes:
        h.add_node(n)

    econ_nodes = [
        HyperNode(
            id="marginal_cost",
            name="Marginal Cost (MC)",
            domains=["Economics"],
            categories=["Property", "Level"],
            aliases=["Costo Marginal"]
        ),
        HyperNode(
            id="marginal_revenue",
            name="Marginal Revenue (MR)",
            domains=["Economics"],
            categories=["Property", "Level"],
            aliases=["Ingreso Marginal"]
        ),
        HyperNode(
            id="profit_maximization",
            name="Profit Maximization Equilibrium (MC = MR)",
            domains=["Economics"],
            categories=["Process", "Evidence"],
            aliases=["Maximización de Beneficios"]
        )
    ]
    for n in econ_nodes:
        h.add_node(n)

    cs_nodes = [
        HyperNode(
            id="algorithmic_complexity",
            name="Algorithmic Complexity & Big-O",
            domains=["ComputerScience"],
            categories=["Property", "Level"],
            aliases=["Complejidad Algorítmica"]
        ),
        HyperNode(
            id="gradient_descent",
            name="Gradient Descent Optimization",
            domains=["ComputerScience"],
            categories=["Process"],
            aliases=["Descenso de Gradiente"]
        ),
        HyperNode(
            id="backpropagation",
            name="Backpropagation & Chain Rule in Neural Networks",
            domains=["ComputerScience"],
            categories=["Process", "Evidence"],
            aliases=["Retropropagación"]
        )
    ]
    for n in cs_nodes:
        h.add_node(n)

    h.add_edge(HyperEdge(
        id="ka_prereq_derivative",
        nodes=["linear_equations", "polynomial_functions", "limits_and_continuity", "derivative_rate_of_change"],
        relation_type="PrerequisiteLattice",
        roles={
            "foundation_slope": "linear_equations",
            "foundation_power": "polynomial_functions",
            "limiting_process": "limits_and_continuity",
            "target_skill": "derivative_rate_of_change"
        },
        domain_scope=["Mathematics"],
        confidence=1.0,
        attributes={"pedagogical_tier": "Calculus 1", "khan_unit": "differentiation-intro"},
        provenance={"source": "Khan Academy Calculus 1 / Differential Calculus"}
    ))

    h.add_edge(HyperEdge(
        id="ka_cross_calculus_to_physics",
        nodes=["derivative_rate_of_change", "kinematic_equations", "instantaneous_velocity"],
        relation_type="CrossDomainBridge",
        roles={
            "math_operator": "derivative_rate_of_change",
            "physical_context": "kinematic_equations",
            "derived_observable": "instantaneous_velocity"
        },
        domain_scope=["Mathematics", "Physics"],
        confidence=0.98,
        attributes={"application": "Position derivative yields velocity; velocity derivative yields acceleration"},
        provenance={"source": "Khan Academy Physics / 1D Kinematics with Calculus"}
    ))

    h.add_edge(HyperEdge(
        id="ka_cross_calculus_to_economics",
        nodes=["derivative_rate_of_change", "marginal_cost", "marginal_revenue", "profit_maximization"],
        relation_type="CrossDomainBridge",
        roles={
            "math_operator": "derivative_rate_of_change",
            "cost_derivative": "marginal_cost",
            "revenue_derivative": "marginal_revenue",
            "equilibrium": "profit_maximization"
        },
        domain_scope=["Mathematics", "Economics"],
        confidence=0.95,
        attributes={"application": "Derivative of Total Cost is MC; Derivative of Total Revenue is MR; d(Profit)/dq = 0"},
        provenance={"source": "Khan Academy Microeconomics / Marginal Analysis"}
    ))

    h.add_edge(HyperEdge(
        id="ka_cross_calculus_to_ai",
        nodes=["derivative_rate_of_change", "vectors_and_spaces", "gradient_descent", "backpropagation", "algorithmic_complexity"],
        relation_type="CrossDomainBridge",
        roles={
            "scalar_derivative": "derivative_rate_of_change",
            "spatial_manifold": "vectors_and_spaces",
            "iterative_algorithm": "gradient_descent",
            "chain_rule_network": "backpropagation",
            "efficiency_constraint": "algorithmic_complexity"
        },
        domain_scope=["Mathematics", "ComputerScience"],
        confidence=0.99,
        attributes={"application": "Multivariate gradients optimize neural weights via reverse-mode automatic differentiation"},
        provenance={"source": "Khan Academy Computer Science / Multivariable Calculus for Deep Learning"}
    ))

    return h
