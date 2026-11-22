"""Build docs/soutenance.pptx (10-minute defense, 16:9) with python-pptx.

Each slide = one message, optional figure, and speaker notes (what to say + timing).
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

DOCS = Path(__file__).resolve().parent
FIG = DOCS.parent / "reports" / "figures"
NAVY, GREY, RED, GREEN = RGBColor(0x0B, 0x3D, 0x6E), RGBColor(0x55, 0x55, 0x55), RGBColor(0xB0, 0x24, 0x24), RGBColor(0x1E, 0x7B, 0x34)

SLIDES: list[dict] = [
    {"title": "Prévision horaire J+1 de la consommation électrique d'un logement",
     "subtitle": "Un protocole sans fuite temporelle\n\nMohamedhen Vall Khaled (21026) · El Moustapha El Moustapha (22013)\n"
                 "AML313 — Séries temporelles de capteurs IoT · Master SupNum · Groupe 1",
     "notes": "(20 s) Présentation. Message : moins un nouveau modèle qu'un protocole vérifiable, sans fuite."},
    {"title": "Le problème métier",
     "bullets": ["Chaque soir à minuit : prévoir les 24 consommations horaires du lendemain",
                 "Usage : piloter un contrat d'énergie (réserver un volume, heures creuses)",
                 "Sous-prévision → achat en urgence, au prix fort · sur-prévision → énergie réservée pour rien",
                 "Deux difficultés : comportement humain peu prévisible, et fuites temporelles faciles"],
     "notes": "(50 s) Insister sur l'asymétrie des coûts : on y reviendra avec le biais de volume."},
    {"title": "Les données et le découpage (figé avant tout modèle)",
     "bullets": ["UCI, logement de Sceaux : 2 075 259 minutes, 12/2006 → 11/2010",
                 "Cible : énergie horaire (kWh) = moyenne des minutes valides × 1 h",
                 "Train → 30/11/2009 · Dev 12/2009-05/2010 · Test 06-11/2010 (scellé)",
                 "1,25 % de minutes manquantes, des lignes entières ; 6 trous > 1 jour = 95 %",
                 "Saisonnalité faible : ρ(24) = 0,43, ρ(168) = 0,46 → hypothèse écrite : gain de 10-20 %"],
     "image": "01_daily_energy_gaps.png",
     "notes": "(60 s) Les trous ne sont pas aléatoires (week-end, août, 2010). L'hypothèse de gain est écrite AVANT les modèles : elle sert de détecteur de fuite."},
    {"title": "Préparation causale : la cible n'est jamais imputée",
     "bullets": ["Heure observée si ≥ 50 min valides (choisi sur le train)",
                 "Une copie remplie de façon causale sert seulement aux retards",
                 "1ʳᵉ heure : dernière valeur · ensuite : médiane des 4 semaines passées",
                 "Interpolation linéaire interdite (utilise le futur) : elle ne gagne que pour les trous ≤ 3 h"],
     "image": "02_fill_methods.png",
     "notes": "(50 s) Testé sur 300 trous simulés par durée. Subtilité : la règle dépend du temps écoulé, car la durée finale d'un trou est inconnue en temps réel."},
    {"title": "Le piège de lag_1 à 24 h d'horizon",
     "bullets": ["y(t−1) par rapport à la cible : inconnu dès h ≥ 2 → FUITE",
                 "Notre lag_1 = dernière valeur connue à minuit (23 h la veille)",
                 "26 variables causales : retards ≥ 24 h, fenêtres arrêtées à minuit, calendrier, fériés",
                 "Test : on modifie tout le jour J → ses variables ne changent pas"],
     "image": "04_lag1_by_horizon.png",
     "notes": "(60 s) Montrer la courbe : la version légale passe de 0,65 à 0,07, la version qui fuit reste à 0,6-0,8. Un modèle qui l'utilise s'effondre en production."},
    {"title": "Protocole : un pare-feu, pas une promesse",
     "bullets": ["Le validateur coupe les données avant chaque origine ; un « modèle espion » le vérifie",
                 "Réapprentissage tous les 28 jours, gap de 24 h",
                 "Réglages sur la fin du train, comparaison sur le dev, mêmes heures pour tous",
                 "IC 95 % par bootstrap par semaines, sur des différences appariées",
                 "Choix final pré-enregistré + verrou SHA-256 → test évalué une seule fois"],
     "notes": "(60 s) C'est le critère à 4 points. Tout est couvert par 35 tests automatisés."},
    {"title": "Baselines : SARIMA, la barre à battre",
     "bullets": ["Dev : naïf-24 0,647 · naïf-168 0,620 · profil 4 sem. 0,517 · SARIMA 0,492 kWh",
                 "SARIMA (1,0,1)(0,1,1)₂₄ : seule la partie saisonnière compte",
                 "Kalman → aucune imputation nécessaire ; résidus non autocorrélés",
                 "Mais l'intervalle gaussien : 97 % la nuit, 64 % à 7 h"],
     "image": "03_mae_coverage_by_horizon.png",
     "notes": "(50 s) La différenciation saisonnière s'adapte au niveau de la saison. Couverture globale de 83 % = moyenne de deux erreurs opposées."},
    {"title": "Ridge et gradient boosting (dev)",
     "bullets": ["Boosting, perte absolue (vise la médiane = optimum de la MAE) : 0,458 kWh",
                 "−6,7 % vs SARIMA, IC [−0,053 ; −0,015] ; meilleur sur les 7 plis",
                 "Ridge : 0,479 (−2,6 %, de justesse)",
                 "Intervalle conforme par heure : couverture 0,72-0,84 à toute heure"],
     "image": "05_mae_coverage_by_horizon.png",
     "notes": "(50 s) Le gain vient de la montée du matin (h = 8 : 0,46 contre 0,75) : le boosting sait si c'est la semaine ou le week-end."},
    {"title": "Ablation : quelle information compte ?",
     "bullets": ["Retards seuls : 0,500 → pas mieux que SARIMA",
                 "+ calendrier : 0,462 → le groupe le plus utile",
                 "Jours fériés : rien de mesurable · lag_1 : seulement h = 1-3 (+9 % sans lui)",
                 "Forte redondance : seule l'heure est irremplaçable ; 8 variables suffiraient",
                 "Fuite volontaire (y(t−1)) : −24 % « gagnés » > tous les vrais gains"],
     "image": "05_ablation.png",
     "notes": "(60 s) La démonstration de fuite justifie a posteriori l'hypothèse écrite au début : un gain spectaculaire doit éveiller le soupçon."},
    {"title": "Test final : évalué une seule fois",
     "table": [["Modèle", "MAE (kWh) [IC 95 %]", "Δ vs SARIMA", "Couverture 80 %"],
               ["Naïf-24", "0,490", "+0,068", "0,840"],
               ["SARIMA", "0,422", "—", "0,811"],
               ["Ridge", "0,416", "−0,006 (ns)", "0,749"],
               ["Boosting", "0,381 [0,337 ; 0,422]", "−0,041 [−0,058 ; −0,024]", "0,768"]],
     "bullets": ["−9,6 % vs SARIMA, −22 % vs naïf : cohérent avec le dev (IC qui se recouvrent)",
                 "Meilleur à toutes les heures et tous les mois ; août : s'adapte aux vacances"],
     "notes": "(70 s) Le test confirme le dev. Ridge ne bat plus SARIMA : on le dit. Les erreurs sont plus faibles qu'au dev parce que l'été consomme moins, pas parce que le modèle s'améliore."},
    {"title": "Limites (rapportées sans correction)",
     "bullets": ["Couverture 76,8 % au lieu de 80 % : la saison change entre calibration et prévision",
                 "Biais de volume : −8,8 % (la médiane est sous la moyenne) → pas pour réserver un volume",
                 "+18 % d'erreur après les longs trous ; pas de température ni de vacances scolaires",
                 "Un seul logement, 2006-2010"],
     "notes": "(50 s) Chaque limite a une piste : conformal adaptatif, perte quadratique pour le volume, données externes."},
    {"title": "Conclusion",
     "bullets": ["Protocole chronologique, testé, pré-enregistré : 0 fuite",
                 "Boosting causal : −9,6 % vs SARIMA, −22 % vs naïf, significatif",
                 "Le gain vient d'une information simple : l'heure × le type de jour",
                 "Un gain spectaculaire est plus souvent une erreur de protocole qu'un meilleur modèle",
                 "Reproductible : python main.py · 35 tests"],
     "notes": "(30 s) Remercier ; ouvrir aux questions."},
]


def _title(slide, text, size=28):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = text; p.font.size = Pt(size); p.font.bold = True; p.font.color.rgb = NAVY
    tb.text_frame.word_wrap = True


def _bullets(slide, items, left, top, width, height, size=17):
    tf = slide.shapes.add_textbox(left, top, width, height).text_frame
    tf.word_wrap = True
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + t; p.font.size = Pt(size); p.space_after = Pt(8)
        if "FUITE" in t or "−24 %" in t:
            p.font.color.rgb = RED


def build_slides() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    for i, s in enumerate(SLIDES, start=1):
        sl = prs.slides.add_slide(blank)
        if "subtitle" in s:
            _title(sl, s["title"], 34)
            sl.shapes[0].top = Inches(2.0)
            tb = sl.shapes.add_textbox(Inches(0.5), Inches(3.6), Inches(12.3), Inches(2.5)).text_frame
            tb.word_wrap = True
            for j, line in enumerate(s["subtitle"].split("\n")):
                p = tb.paragraphs[0] if j == 0 else tb.add_paragraph()
                p.text = line; p.font.size = Pt(18 if j == 0 else 15); p.font.color.rgb = GREY
        else:
            _title(sl, s["title"])
            img = s.get("image")
            if "table" in s:
                rows, cols = len(s["table"]), len(s["table"][0])
                tbl = sl.shapes.add_table(rows, cols, Inches(0.7), Inches(1.4), Inches(11.9), Inches(2.6)).table
                for r, row in enumerate(s["table"]):
                    for c, val in enumerate(row):
                        cell = tbl.cell(r, c); cell.text = val
                        para = cell.text_frame.paragraphs[0]; para.font.size = Pt(16); para.font.bold = (r == 0 or r == rows - 1)
                        if r == rows - 1:
                            para.font.color.rgb = GREEN
                _bullets(sl, s["bullets"], Inches(0.7), Inches(4.4), Inches(11.9), Inches(2.5))
            elif img and (FIG / img).exists():
                _bullets(sl, s["bullets"], Inches(0.5), Inches(1.3), Inches(5.6), Inches(5.8), size=16)
                pic = sl.shapes.add_picture(str(FIG / img), Inches(6.3), Inches(1.4))
                ratio = pic.height / pic.width
                pic.width = Inches(6.7); pic.height = Emu(int(pic.width * ratio))
                if pic.height > Inches(5.6):
                    pic.height = Inches(5.6); pic.width = Emu(int(pic.height / ratio))
            else:
                _bullets(sl, s["bullets"], Inches(0.7), Inches(1.4), Inches(11.9), Inches(5.6), size=20)
        foot = sl.shapes.add_textbox(Inches(11.8), Inches(7.0), Inches(1.3), Inches(0.4)).text_frame.paragraphs[0]
        foot.text = f"{i} / {len(SLIDES)}"; foot.font.size = Pt(11); foot.font.color.rgb = GREY
        sl.notes_slide.notes_text_frame.text = s["notes"]
    out = DOCS / "soutenance.pptx"
    prs.save(out)
    print(f"soutenance.pptx: {len(SLIDES)} slides")
    return out


if __name__ == "__main__":
    build_slides()
