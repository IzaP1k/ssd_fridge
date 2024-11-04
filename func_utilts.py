import torch


def find_jaccard_overlap(boxes, priors):
    """
    Oblicza IoU (Intersection over Union) między każdą parą boxów i priorów.

    :param boxes: Tensor o wymiarach (n_object, 4), gdzie każdy box ma format [xmin, ymin, xmax, ymax]
    :param priors: Tensor o wymiarach (n_p, 4), gdzie każdy prior ma format [xmin, ymin, xmax, ymax]
    :return: Tensor o wymiarach (n_object, n_p), reprezentujący IoU dla każdej pary (box, prior)
    """
    # Rozdziel koordynaty dla boxes i priors
    boxes_xmin, boxes_ymin, boxes_xmax, boxes_ymax = boxes[:, 0].unsqueeze(1), boxes[:, 1].unsqueeze(1), boxes[:,
                                                                                                         2].unsqueeze(
        1), boxes[:, 3].unsqueeze(1)
    priors_xmin, priors_ymin, priors_xmax, priors_ymax = priors[:, 0].unsqueeze(0), priors[:, 1].unsqueeze(0), priors[:,
                                                                                                               2].unsqueeze(
        0), priors[:, 3].unsqueeze(0)

    # Oblicz współrzędne skrzyżowań (intersekcji)
    inter_xmin = torch.max(boxes_xmin, priors_xmin)
    inter_ymin = torch.max(boxes_ymin, priors_ymin)
    inter_xmax = torch.min(boxes_xmax, priors_xmax)
    inter_ymax = torch.min(boxes_ymax, priors_ymax)

    # Oblicz szerokość i wysokość intersekcji, upewnij się, że wartości są nieujemne
    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_width * inter_height  # (n_object, n_p)

    # Oblicz powierzchnię boxów i priorów
    boxes_area = (boxes_xmax - boxes_xmin) * (boxes_ymax - boxes_ymin)  # (n_object, 1)
    priors_area = (priors_xmax - priors_xmin) * (priors_ymax - priors_ymin)  # (1, n_p)

    # Oblicz powierzchnię sumaryczną
    union_area = boxes_area + priors_area - inter_area

    # Oblicz IoU, dodaj małą wartość do mianownika, aby uniknąć dzielenia przez zero
    iou = inter_area / (union_area + 1e-6)

    return iou  # (n_object, n_p)

def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - cxcy[:, 2:] / 2,
                      cxcy[:, :2] + cxcy[:, 2:] / 2], dim=1)


def xy_to_cxcy(boxes):
    """
    Przekształć współrzędne prostokątne z formatu (xmin, ymin, xmax, ymax)
    na format (cx, cy, w, h).

    :param boxes: tensor o rozmiarze [N_objects, 4] z współrzędnymi (xmin, ymin, xmax, ymax)
    :return: tensor o rozmiarze [N_objects, 4] w formacie (cx, cy, w, h)
    """
    cxcy = torch.empty_like(boxes)
    cxcy[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
    cxcy[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
    cxcy[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
    cxcy[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
    return cxcy


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Przekształć współrzędne (cx, cy, w, h) na skumulowaną reprezentację względem priorytetów.

    :param cxcy: tensor o rozmiarze [N_objects, 4] w formacie (cx, cy, w, h)
    :param priors_cxcy: tensor o rozmiarze [2040, 4] z priorytetami (cx, cy, w, h)
    :return: tensor o rozmiarze [N_objects, 4] w formacie (gcx, gcy, gw, gh)
    """
    # Powtórzenie priors_cxcy do dopasowania wymiaru do cxcy
    num_repeats = cxcy.size(0) // priors_cxcy.size(0) + 1  # Dodajemy 1, aby mieć wystarczającą ilość priorytetów
    priors_cxcy_expanded = priors_cxcy.repeat(num_repeats, 1)[:cxcy.size(0), :]  # Rozszerzamy do [179580, 4] i tniemy

    # Obliczenia gcxgcy z dopasowanym wymiarem
    gcxgcy = torch.empty_like(cxcy)
    gcxgcy[:, :2] = (cxcy[:, :2] - priors_cxcy_expanded[:, :2]) / priors_cxcy_expanded[:, 2:]  # (cx, cy)
    gcxgcy[:, 2:] = torch.log(cxcy[:, 2:] / priors_cxcy_expanded[:, 2:])  # (w, h)

    return gcxgcy
