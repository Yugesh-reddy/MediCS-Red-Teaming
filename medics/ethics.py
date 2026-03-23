"""
MediCS — Ethical Framework
===========================
Version-controlled ethics statement template covering dual-use risks,
responsible disclosure, and broader impact. Generates a structured
ethics section suitable for paper appendices or supplementary materials.
"""


def generate_ethics_statement(
    model_name="Llama-3-8B-Instruct",
    n_seeds=500,
    n_languages=6,
    attack_strategies=5,
):
    """
    Generate a structured ethics statement for the MediCS paper.

    Returns:
        dict with keys: dual_use, responsible_disclosure, broader_impact,
        data_ethics, limitations, mitigations
    """
    return {
        "dual_use": {
            "risk": (
                "This work demonstrates that code-switching between English and "
                "low-resource languages can bypass safety filters in medical LLMs. "
                "The attack prompts and strategies could be misused to extract "
                "harmful medical information from deployed systems."
            ),
            "justification": (
                "We follow the established responsible disclosure paradigm in "
                "adversarial ML research. Identifying vulnerabilities is necessary "
                "to develop defenses. Our iterative SFT+DPO defense framework "
                "directly addresses the vulnerabilities we expose, and we release "
                "the defense alongside the attack methodology."
            ),
        },
        "responsible_disclosure": {
            "dataset": (
                f"MediCS-500 contains {n_seeds} adversarial medical prompts across "
                f"{n_languages} languages and {attack_strategies} attack strategies. "
                "We release the dataset with access controls on HuggingFace Hub, "
                "requiring users to agree to a responsible use license before download."
            ),
            "model": (
                f"We release LoRA defense adapters for {model_name}, not the full "
                "merged model. This allows other researchers to reproduce our defense "
                "without distributing a model that could be misused without the adapter."
            ),
            "code": (
                "Attack generation code is released to enable reproducibility and "
                "extension. The code requires API keys and computational resources, "
                "providing a practical barrier to casual misuse."
            ),
        },
        "broader_impact": {
            "positive": [
                "Demonstrates that low-resource language speakers face disproportionate "
                "safety risks from current LLMs — raising awareness for equitable AI safety.",
                "Provides a practical, low-cost defense framework ($10 API budget) "
                "that smaller organizations can adopt.",
                "Contributes to the understanding of multilingual safety alignment, "
                "an under-researched area critical for global AI deployment.",
            ],
            "negative": [
                "Attack strategies could be adapted to non-medical domains.",
                "Defense may create false confidence — our method reduces but does not "
                "eliminate vulnerability.",
                "Focus on 6 languages leaves many low-resource languages unexamined.",
            ],
        },
        "data_ethics": {
            "no_real_patients": (
                "All seed prompts are synthetically generated. No real patient data, "
                "medical records, or personally identifiable information is used."
            ),
            "harmful_content": (
                "Seed prompts describe harmful scenarios (drug synthesis, self-harm "
                "methods, etc.) for the purpose of testing safety filters. These are "
                "modeled on categories from established safety benchmarks (AdvBench, "
                "HarmBench) and medical ethics literature."
            ),
            "language_representation": (
                f"We selected {n_languages} languages across 3 script families and "
                "3 language families to maximize linguistic diversity within budget "
                "constraints. We acknowledge this is not comprehensive."
            ),
        },
        "limitations": [
            f"Single target model ({model_name}) — defense generalization unverified.",
            "Automated judge (GPT-4o) may have systematic biases in multilingual evaluation.",
            "Code-switching is one of many possible multilingual attack vectors.",
            "Defense effectiveness may degrade against adaptive adversaries.",
            "Low-resource translation quality varies — some code-switched prompts may be unnatural.",
        ],
        "mitigations": [
            "Defense released alongside attack methodology.",
            "Dataset gated behind responsible use license.",
            "Transfer evaluation (Mistral-7B) tests cross-model generalization.",
            "Back-translation verification ensures code-switched prompts preserve semantics.",
            "Multi-seed evaluation (3 seeds) with bootstrap CIs for statistical rigor.",
        ],
    }


def format_ethics_section(ethics_dict):
    """
    Format the ethics statement dictionary as a text section
    suitable for inclusion in a paper appendix.

    Args:
        ethics_dict: output of generate_ethics_statement()

    Returns:
        str: formatted ethics section text
    """
    lines = []
    lines.append("ETHICS STATEMENT")
    lines.append("=" * 60)

    # Dual Use
    lines.append("\n1. DUAL-USE CONSIDERATIONS")
    lines.append(f"   Risk: {ethics_dict['dual_use']['risk']}")
    lines.append(f"   Justification: {ethics_dict['dual_use']['justification']}")

    # Responsible Disclosure
    lines.append("\n2. RESPONSIBLE DISCLOSURE")
    for key, val in ethics_dict["responsible_disclosure"].items():
        lines.append(f"   {key.capitalize()}: {val}")

    # Broader Impact
    lines.append("\n3. BROADER IMPACT")
    lines.append("   Positive:")
    for item in ethics_dict["broader_impact"]["positive"]:
        lines.append(f"     - {item}")
    lines.append("   Negative:")
    for item in ethics_dict["broader_impact"]["negative"]:
        lines.append(f"     - {item}")

    # Data Ethics
    lines.append("\n4. DATA ETHICS")
    for key, val in ethics_dict["data_ethics"].items():
        lines.append(f"   {key}: {val}")

    # Limitations
    lines.append("\n5. LIMITATIONS")
    for item in ethics_dict["limitations"]:
        lines.append(f"   - {item}")

    # Mitigations
    lines.append("\n6. MITIGATIONS")
    for item in ethics_dict["mitigations"]:
        lines.append(f"   - {item}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
