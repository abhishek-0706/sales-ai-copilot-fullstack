"""Static checks for deployment artefacts (no cluster or Docker needed).

Parses every YAML file and checks: required Kubernetes fields, Services select an existing
Deployment, the HPA targets an existing Deployment, containers declare probes, resources and a
non-root security context, kustomization resources exist, compose and CI files parse.
Usage: python scripts/validate_manifests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "deploy" / "k8s"


def main() -> int:
    errors: list[str] = []
    docs: list[tuple[str, dict]] = []
    for path in sorted(K8S.glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc:
                docs.append((path.name, doc))
    for name, d in docs:
        for key in ("apiVersion", "kind"):
            if key not in d:
                errors.append(f"{name}: missing {key}")
        if d.get("kind") != "Kustomization" and not d.get("metadata", {}).get("name"):
            errors.append(f"{name}: missing metadata.name")

    deployments = {d["metadata"]["name"]: d for _, d in docs if d.get("kind") == "Deployment"}
    for name, d in deployments.items():
        spec = d["spec"]
        if not spec["selector"]["matchLabels"].items() <= spec["template"]["metadata"]["labels"].items():
            errors.append(f"Deployment {name}: selector does not match template labels")
        if not spec["template"]["spec"].get("securityContext", {}).get("runAsNonRoot"):
            errors.append(f"Deployment {name}: runAsNonRoot not set")
        for c in spec["template"]["spec"]["containers"]:
            for key in ("readinessProbe", "livenessProbe", "resources"):
                if key not in c:
                    errors.append(f"Deployment {name}/{c['name']}: missing {key}")
    for _, d in docs:
        if d.get("kind") == "Service":
            sel = d["spec"]["selector"]
            if not any(sel.items() <= dep["spec"]["template"]["metadata"]["labels"].items()
                       for dep in deployments.values()):
                errors.append(f"Service {d['metadata']['name']}: selector matches no Deployment")
        if d.get("kind") == "HorizontalPodAutoscaler":
            target = d["spec"]["scaleTargetRef"]["name"]
            if target not in deployments:
                errors.append(f"HPA {d['metadata']['name']}: target {target} not found")
        if d.get("kind") == "Kustomization":
            for res in d.get("resources", []):
                if not (K8S / res).exists():
                    errors.append(f"kustomization: missing resource {res}")

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    if set(compose.get("services", {})) != {"api", "web"}:
        errors.append("docker-compose.yml: expected services api and web")
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    if not ci.get("jobs"):
        errors.append("ci.yml: no jobs")

    kinds = sorted(f"{d['kind']}/{d.get('metadata', {}).get('name', '-')}" for _, d in docs)
    print(f"parsed {len(docs)} Kubernetes documents: {', '.join(kinds)}")
    print(f"parsed docker-compose.yml ({len(compose['services'])} services) and ci.yml ({len(ci['jobs'])} jobs)")
    for e in errors:
        print("ERROR", e)
    print("manifests OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
