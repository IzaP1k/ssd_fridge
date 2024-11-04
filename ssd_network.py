import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from pathlib import Path as p
from datetime import datetime

import func_utilts as utils
import general_utils


basic_conv = nn.Conv2d  # change this to 3d if needed

# WAŻNE W PRZYPADKU CONV - MUSI BYC PARZYSTE

class Pad_to_even_size(nn.Module):
    def __init__(self):
        super(Pad_to_even_size, self).__init__()

    def forward(self, x):
        a = []
        for i in x.shape[2:]:  # pomijamy batch i kanał
            if i % 2 == 1:
                a = [0, 1] + a  # Dodaj padding 1 dla nieparzystych wymiarów
            else:
                a = [0, 0] + a  # Dodaj padding 0 dla parzystych wymiarów
        return torch.nn.functional.pad(x, a) # Funkcja ta nakłada padding na tensor x, zgodnie z wartościami a, co zapewnia, że wyjście ma wymiary parzyste.


def pretty_print_module_list(module_list, x, print_net=True, max_colwidth=500):
    '''x: dummyinput [batch=1, C, H, W]
    Wejście module_list: Lista modułów (warstw), które będą przetwarzane sekwencyjnie.
x: Tensor wejściowy, który jest przepuszczany przez każdy moduł.
Tworzenie DataFrame df:
Zawiera nazwę warstwy w kolumnie Layer.
W kolumnie Output size przechowuje rozmiar wyjściowy po przetworzeniu przez każdą warstwę.
Pętla for: Przetwarza tensor przez kolejne warstwy i rejestruje jego rozmiar wyjściowy.
Wydruk: Jeśli print_net jest True, funkcja wypisuje tabelę z warstwami i ich rozmiarami wyjściowymi.
    '''
    pd.options.display.max_colwidth = max_colwidth
    df = pd.DataFrame({'Layer': list(map(str, module_list))})
    output_size = []
    for i, layer in enumerate(module_list):
        x = layer(x)
        output_size.append(tuple(x.size()))
    df['Output size'] = output_size
    if print_net:
        print('\n', df, '\n')
    return df['Output size'].tolist()


# ========================conv======================
"""
Parametry Conv:
inC, outC: Liczba kanałów wejściowych i wyjściowych.

kernel_size, padding, stride, groups: Standardowe parametry dla warstwy konwolucyjnej.

spectral: Jeśli ustawione na True, stosowana jest normalizacja spektralna (spectral_norm), co stabilizuje uczenie i ogranicza wartości własne wag.

Funkcja basic_conv: Domyślnie warstwa konwolucyjna, którą Conv konstruuje i ewentualnie normalizuje spektralnie.
"""
# Normalizacja spektralna: Jeśli parametr spectral jest ustawiony na True, stosowana jest spectral_norm,
# co pomaga w stabilizacji uczenia, ograniczając wartości własne wag. Taka normalizacja zapobiega zbyt dużym zmianom
# wartości aktywacji, co może poprawić dokładność sieci.
class Conv(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, padding=1, stride=1, groups=1, spectral=False):
        super(Conv, self).__init__()
        if spectral:
            self.conv = spectral_norm(
                basic_conv(inC, outC, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups))
        else:
            self.conv = basic_conv(inC, outC, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups)

    def forward(self, x):
        x = self.conv(x)
        return x


# =======================norm======================
"""
Bez normalizacji: Ta klasa działa jak "pusta" warstwa, która po prostu przepuszcza tensor bez zmian. Może być użyta zamiast normalizacji, jeśli potrzebujemy zdefiniować jednolitą architekturę, gdzie w pewnych miejscach normalizacja nie jest potrzebna.
Parametr c: Parametr c nie jest używany i prawdopodobnie służy do zapewnienia zgodności interfejsu klasy No_norm z innymi normalizacjami.
"""
class No_norm(nn.Module):
    def __init__(self, c):
        super(No_norm, self).__init__()

    def forward(self, x):
        return x

def init_param(m):
    """
    Initialize convolution parameters.
    bazowe parametry a i b
    """
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

# EKSTRAKCJA CECH
class BaseConv(nn.Module):
    def __init__(self, conv_layers, input_size, chosen_fm=[-1, -2],
                 norm=nn.InstanceNorm2d, conv=Conv, act_fn=nn.LeakyReLU(0.01), spectral=False):
        '''
        conv_layers: list of channels from input to output
        for example, conv_layers=[1,10,20]
        --> module_list=[
            conv(1,10), norm, act_fn,
            conv(10,10), norm, act_fn,
            maxpool,
            conv(10, 20), norm, act_fn,
            conv(20, 20), norm, act_fn]

        input_size: int (assume square images)

        conv_layers: Lista kanałów dla każdej warstwy konwolucyjnej, np. [1,10,20]. Oznacza to, że każda kolejna warstwa konwolucyjna będzie mieć odpowiednią liczbę filtrów (np. 10 lub 20).
    input_size: Rozmiar wejściowego obrazu; zakłada się, że obrazy są kwadratowe.
    chosen_fm: Lista indeksów wybranych map cech (feature maps) używanych do predykcji. Domyślnie są to ostatnie dwie.
    norm, conv, act_fn, spectral: Parametry ustawiające odpowiednie funkcje normalizacji, konwolucji, aktywacji oraz opcjonalnie normalizację spektralną, która stabilizuje gradienty.
        '''
        super(BaseConv, self).__init__()
        #create module list
        self.module_list = nn.ModuleList()
        self.fm_id = []
        for i in range(len(conv_layers)-1):
            # dopasowujemy wymiar do parzystego
            if input_size % 2 == 1:
                self.module_list.append(Pad_to_even_size())
            self.module_list.extend([
                conv(inC=conv_layers[i], outC=conv_layers[i+1], spectral=spectral),
                norm(conv_layers[i+1]),  # normalizacja
                act_fn,  # dodaje nieliniowość - leaky relu
                conv(inC=conv_layers[i+1], outC=conv_layers[i+1], spectral=spectral),
                norm(conv_layers[i+1]),
                act_fn,
                nn.MaxPool2d(kernel_size=2)]
            )
            input_size = np.ceil(input_size / 2)  # zmniejszamy rozmiar przy poolingu

            # select feature maps for prediction. They are the output of act_fn right before maxpool layers
            self.fm_id.append(len(self.module_list) - 2)

        self.fm_id = [self.fm_id[i] for i in chosen_fm]  # only use the last 2 fm in base conv

        self.module_list = self.module_list[:-1]  # ignore last maxpool layer, max rozdzielczość

        self.output_size = input_size

    def forward(self, x):
        fm = []
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
            if i in self.fm_id:
                fm.append(x)
        return x, fm # przetworzone dane i wybrane cechy


class AuxConv(nn.Module):
    def __init__(self, conv_layers, input_size,
                 norm=nn.InstanceNorm2d, conv=Conv, act_fn=nn.LeakyReLU(0.01), spectral=False):
        '''
        conv_layers: list of channels from input to output.
        for example, conv_layers=[c1,c2,c3]
        --> module_list=[
            conv(c1, c1//2, kernel_size=1, pad=0), act_fn,
            conv(c1//2, c2, 3, 1, stride=2), act_fn,
            conv(c2, c2//2), norm, act_fn,
            conv(c2//2, c3), norm, act_fn]

        input_size: int (assume square images)
        '''
        super(AuxConv, self).__init__()
        self.module_list = nn.ModuleList()
        self.fm_id = []
        for i in range(len(conv_layers) - 1):
            self.module_list.extend(
                [conv(conv_layers[i], conv_layers[i] // 2, kernel_size=1, padding=0, spectral=spectral),
                 norm(conv_layers[i] // 2),
                 act_fn,
                 conv(conv_layers[i] // 2, conv_layers[i + 1], kernel_size=3, padding=1, stride=2, spectral=spectral),
                 norm(conv_layers[i + 1]),
                 act_fn])
            """
            Warstwa konwolucji 1x1: Zmniejsza liczbę kanałów o połowę, co umożliwia kompresję danych bez znaczącej utraty informacji. Kernel 1x1 jest efektywny i zmniejsza liczbę obliczeń.
            
            Normalizacja i aktywacja: Zastosowanie funkcji aktywacji i normalizacji pozwala lepiej reprezentować informacje i unikać problemów z gradientami.
            
            Warstwa konwolucji 3x3: Używa jądra 3x3 i zwiększa rozmiar kanałów (stride=2 zmniejsza rozdzielczość map cech).
            
            Dodanie fm_id: Na końcu każdej iteracji dodawany jest indeks, aby zapisywać wyjścia map cech na różnych etapach przetwarzania.
            """
            self.fm_id.append(len(self.module_list) - 1)

    def forward(self, x):
        fm = []
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
            if i in self.fm_id:
                fm.append(x)
        return fm


class PredictionConv(nn.Module):
    def __init__(self, n_classes, fm_channels, n_prior_per_pixel, conv=Conv, norm=nn.InstanceNorm2d, spectral=False):
        """

        n_classes: Liczba klas obiektów, które model ma rozpoznawać (np. samochód, pies).
        fm_channels: Lista kanałów dla każdej mapy cech, które zostały wygenerowane wcześniej przez bloki konwolucyjne.
        n_prior_per_pixel: Lista liczby „priors” (z ang. „proponowanych okien granicznych”) na każdy piksel, które model rozważa jako potencjalne obszary dla obiektów.
        conv, norm, spectral: Parametry definiujące rodzaj warstw konwolucyjnych i normalizacyjnych oraz opcjonalną normalizację spektralną.

        :param n_classes:
        :param fm_channels:
        :param n_prior_per_pixel:
        :param conv:
        :param norm:
        :param spectral:
        """
        super(PredictionConv, self).__init__()
        self.n_classes = n_classes

        # localization conv out layers
        self.loc_module_list = nn.ModuleList()
        for i in range(len(fm_channels)):
            self.loc_module_list.append(nn.Sequential(norm(fm_channels[i]),
                                                      conv(fm_channels[i], n_prior_per_pixel[i] * 4, kernel_size=3,
                                                           padding=1, spectral=spectral)))
        """
        self.loc_module_list: Tworzy listę modułów, które będą przewidywać lokalizacje granic obiektów.
        
        Pętla for i in range(len(fm_channels)): Dla każdej mapy cech tworzy osobny moduł konwolucyjny.
        norm(fm_channels[i]): Normalizacja mapy cech w celu stabilizacji treningu.
        conv: Warstwa konwolucyjna z liczbą filtrów równą n_prior_per_pixel[i] * 4, co odpowiada liczbie współrzędnych dla każdego okna (x, y, szerokość, wysokość).
        """
        # prediction conv out layers
        self.cla_module_list = nn.ModuleList()
        for i in range(len(fm_channels)):
            self.cla_module_list.append(nn.Sequential(norm(fm_channels[i]),
                                                      conv(fm_channels[i], n_prior_per_pixel[i] * n_classes,
                                                           kernel_size=3, padding=1, spectral=spectral)))
        """
        self.cla_module_list: Lista modułów, które przewidują klasy obiektów na mapach cech.
        Parametry warstw:
        norm(fm_channels[i]): Normalizacja kanałów cech.
        conv: Liczba filtrów równa n_prior_per_pixel[i] * n_classes (dla każdej klasy, która może być przypisana priors w danym pikselu).
        """

    def postprocess(self, x, k):
        '''x: output of self.(loc/cla)module_list. size [batch, n_boxes*k, h, w]. reshape into [batch, n_boxes*h*w, k]
           k: 4 or n_classes

           Argumenty:
            x: Wyjście modułów lokalizacji lub klasyfikacji (w formacie [batch, liczba_priors*k, h, w]).
            k: Dla loc_module_list wynosi 4 (x, y, szerokość, wysokość); dla cla_module_list wynosi n_classes.
            Permutacja permute([0, 2, 3, 1]): Zmienia kolejność wymiarów z [batch, channels, height, width] na [batch, height, width, channels].
            Zmiana wymiarów view(x.size(0), -1, k): Rozwija dane do wymiarów [batch, total_priors, k], gdzie total_priors to całkowita liczba priors.
           '''
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = x.view(x.size(0), -1, k)
        return x

    def forward(self, fm):
        '''
        fm: list[n_fm] of torch tensors[batch, channel, h,w]: feature maps that contain priors
        return: loc_output[]
        '''
        loc_output = []
        cla_output = []

        for i in range(len(fm)):
            """
            Dla każdej mapy cech (fm[i]):
            Przetwarza mapę cech przez odpowiedni moduł z loc_module_list w celu uzyskania prognozy lokalizacji (wyjście 4).
            Przetwarza mapę cech przez odpowiedni moduł z cla_module_list w celu uzyskania prognozy klasyfikacji (self.n_classes).
            self.postprocess: Formatowanie wyjścia w postaci [batch, total_priors, 4] dla lokalizacji oraz [batch, total_priors, n_classes] dla klasyfikacji.
            """
            loc_output.append(self.postprocess(self.loc_module_list[i](fm[i]), 4))

            cla_output.append(self.postprocess(self.cla_module_list[i](fm[i]), self.n_classes))

        """
        Łączenie wzdłuż osi dim=1: Wszystkie przewidywania lokalizacyjne i klasyfikacyjne są łączone, tworząc pojedyncze wyjścia dla lokalizacji [batch, total_n_prior, 4] oraz klasyfikacji [batch, total_n_prior, n_classes].
        
        Zwracane wartości:
        loc_output: Współrzędne okien granicznych dla obiektów.
        cla_output: Klasyfikacja dla każdego obiektu w oknach.
        """

        loc_output = torch.cat(loc_output, dim=1)  # [batch, total_n_prior, 4]
        cla_output = torch.cat(cla_output, dim=1)  # [batch, total_n_prior, n_classes]

        return loc_output, cla_output

class SSD(nn.Module):
    def __init__(self, config, base_conv=None, aux_conv=None):
        """

        Warstwa Bazowa Konwolucyjna (base_conv): Odpowiedzialna za ekstrakcję głównych cech z obrazu wejściowego.
        Warstwa Pomocnicza Konwolucyjna (aux_conv): Dodatkowe warstwy do ekstrakcji cech, które są przydatne do generowania wyższych poziomów map cech.
        Warstwa Predykcyjna (pred_conv): Używa klasy PredictionConv do generowania lokalizacji i klasyfikacji dla każdego prior boxa.
        Prior Boksy (priors_cxcy): Boksy bazowe, wygenerowane przez funkcję create_prior_boxes(), które stanowią domyślne ramki otaczające, służące do wykrywania obiektów w różnych skalach i proporcjach.

        :param config:
        :param base_conv:
        :param aux_conv:
        """
        super(SSD, self).__init__()
        self.config = config

        if base_conv != None:
            self.base_conv = base_conv
        else:
            self.base_conv = BaseConv(self.config['base_conv_conv_layers'], self.config['base_conv_input_size'],
                                      norm=No_norm)

        if aux_conv != None:
            self.aux_conv = aux_conv
        else:
            self.aux_conv = AuxConv(self.config['aux_conv_conv_layers'], self.config['aux_conv_input_size'],
                                    norm=No_norm)

        self.pred_conv = PredictionConv(self.config['n_classes'], self.config['fm_channels'],
                                        self.config['n_prior_per_pixel'])

        # prior boxes
        self.priors_cxcy = self.create_prior_boxes()
        self.n_p = len(self.priors_cxcy)

        self.apply(init_param)
        print('Done initialization')

    def forward(self, x):
        '''

        x przechodzi przez base_conv, co generuje główną mapę cech oraz listę map cech (fm) z warstw pośrednich.
        Do listy fm dodawane są dodatkowe mapy cech z aux_conv.
        pred_conv przetwarza fm, generując loc_output oraz cla_output, czyli lokalizacje i klasyfikacje dla każdego prior boxa.
        Funkcja zwraca loc_output, cla_output oraz fm.

        x: tensor[N, 3, 300, 300]
        returns predictions:
            loc_output (N, n_p, 4)
            cla_output (N, n_p, n_classes)
        '''
        x, fm = self.base_conv(x)

        fm.extend(self.aux_conv(x))

        loc_output, cla_output = self.pred_conv(fm)
        return loc_output, cla_output, fm

    def create_prior_boxes(self):
        '''
        input: self.config['fm_size',
        'fm_prior_scale', 'fm_prior_aspect_ratio']

        return: prior boxes in center-size coordinates.
        Tensor size [n_p, 4]

        Ta metoda generuje priory boksy (domyślne ramki otaczające) w układzie "center-size" dla map cech. Funkcja iteruje przez wymiary map cech oraz skale, aby stworzyć ramki o różnych proporcjach, co umożliwia detekcję obiektów o różnych rozmiarach i kształtach.

        Struktura:
        Funkcja iteruje przez każdą mapę cech, definiując jej wymiary (d), skale (s) oraz proporcje (r) na podstawie ustawień konfiguracyjnych.
        Oblicza centrum (cx, cy) oraz wymiary każdej ramki.
        Lista ramek jest konwertowana na torch.FloatTensor, wartości są ograniczane do zakresu 0–1, a wynik jest zwracany.

        '''
        priors = []
        for i in range(self.config['n_fm']):
            d = self.config['fm_size'][i]
            s = self.config['fm_prior_scale'][i]
            for j in range(d):
                for k in range(d):
                    # Note the order of k, j vs x,y here. It must be consistent with the permute/view operation in PredictionConv.post_process_output
                    cx = (k + 0.5) / d
                    cy = (j + 0.5) / d
                    for r in self.config['fm_prior_aspect_ratio'][i]:
                        priors.append([cx, cy, s * np.sqrt(r), s / np.sqrt(r)])
                        if r == 1:
                            try:
                                additional_scale = np.sqrt(s * self.config['fm_prior_scale'][i + 1])
                            except IndexError:
                                additional_scale = 1.
                            priors.append([cx, cy, additional_scale, additional_scale])
        priors = torch.FloatTensor(priors).to(self.config['device'])
        priors.clamp_(0, 1)
        print(f"There are {len(priors)} priors in this model")
        return priors

    def detect_object(self, loc_output, cla_output, min_score, max_overlap, top_k):
        '''
        loc_output: size [n, n_p, 4]
        cla_output: size [n, n_p, n_classes]

        return:


        Parametry:

        loc_output: Przewidywane lokalizacje ramek otaczających.
        cla_output: Przewidywane wyniki klas.
        min_score: Minimalny próg wyniku dla uznania detekcji.
        max_overlap: Maksymalne dopuszczalne nakładanie się ramek w algorytmie NMS (Non-Maximum Suppression).
        top_k: Maksymalna liczba ramek do zachowania.
        Przepływ:

        cla_output przechodzi przez softmax, aby uzyskać prawdopodobieństwa dla każdej klasy.
        Dla każdego obrazu w batchu, funkcja dekoduje lokalizacje do ramek otaczających, stosuje NMS do redukcji nakładania się ramek i filtruje na podstawie wyniku.
        Dodaje końcowe detekcje dla każdego obrazu w batchu, w tym ramki otaczające, etykiety klas i wyniki.
        '''
        # print('detecting...')
        batch_size = loc_output.size(0)

        cla_output = F.softmax(cla_output, dim=2)  # [N, N_P, n_classes]
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        for i in range(batch_size):
            decoded_locs = utils.cxcy_to_xy(
                utils.gcxgcy_to_cxcy(loc_output[i], self.priors_cxcy))  # (n_p, 4), fractional pt. coord
            #
            image_boxes = []
            image_labels = []
            image_scores = []

            max_score, best_label = cla_output[i].max(dim=1)
            for c in range(self.config['n_classes'] - 1):
                """
                
                cla_output[i][:, c]: Pobiera wyniki dla klasy c dla wszystkich priorytetowych ramek (ang. prior boxes).
                above_min_score_index: Tworzy maskę logiczną wskazującą, które wyniki są wyższe od min_score.
                class_score = class_score[above_min_score_index]: Filtruje wyniki, pozostawiając tylko te, które spełniają próg min_score.

                """
                class_score = cla_output[i][:, c]  # [n_p]
                above_min_score_index = (class_score > min_score)  # [n_p]
                class_score = class_score[above_min_score_index]  # [n_p_min]

                """
                
                Jeśli żaden wynik nie przekroczył min_score, przechodzi do kolejnej klasy.
                sorted_score i sorted_index: Sortuje wyniki w malejącej kolejności, co pozwala później na wybór ramek o najwyższych wynikach dla tej klasy.
                
                """

                if len(class_score) == 0:
                    continue
                sorted_score, sorted_index = class_score.sort(dim=0, descending=True)  # [n_p_min]

                # print('decoded_locs.size() = ', decoded_locs.size(), above_min_score_index.size(), sorted_index.size())

                """
                keep: Inicjalizuje maskę logiczną (keep), ustawiając wszystkie wartości na 1, co oznacza, że początkowo wszystkie ramki są zachowane.
                iou: Oblicza macierz Jaccarda (IoU) między wszystkimi parami ramek otaczających po posortowaniu. Wynik to macierz [n_p_min, n_p_min], gdzie każda wartość określa stopień nakładania się między dwoma ramkami.
                """

                keep = torch.ones_like(sorted_score, dtype=torch.uint8).to(self.config['device'])  # [n_p_min]
                iou = utils.find_jaccard_overlap(decoded_locs[above_min_score_index][sorted_index],
                                                 decoded_locs[above_min_score_index][
                                                     sorted_index])  # [n_p_min, n_p_min]

                """
                
                Pętla sprawdza każdą ramkę w sorted_index. Jeśli keep[j] wynosi 1, oznacza to, że ta ramka jest zachowana. Wtedy kod ustawia wartości keep na 0 dla ramek, które mają zbyt duże nakładanie się (IoU > max_overlap) w stosunku do aktualnie wybranej ramki.
                Proces ten umożliwia zachowanie tylko najważniejszych, nie-nakładających się ramek dla danej klasy.
                
                """

                # print(utils.rev_label_map[c], iou)
                for j in range(len(sorted_index) - 1):
                    if keep[j] == 1:
                        keep[j + 1:] = torch.min(keep[j + 1:], iou[j, j + 1:] <= max_overlap)
                # print(utils.find_jaccard_overlap(decoded_locs[above_min_score_index][sorted_index][keep],
                #                                  decoded_locs[above_min_score_index][sorted_index][keep])) #[n_p_min, n_p_min])


                """
                
                image_boxes: Dodaje wybrane ramki otaczające dla klasy c, które przeszły przez NMS.
                image_labels: Dodaje etykiety klasy c dla każdej wybranej ramki.
                image_scores: Dodaje odpowiednie wyniki.
                
                """

                image_boxes.append(decoded_locs[above_min_score_index][sorted_index][keep])
                image_labels += [c] * keep.sum()
                image_scores.append(sorted_score[keep])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.config['device']))
                image_labels.append(torch.LongTensor([0]).to(self.config['device']))
                image_scores.append(torch.FloatTensor([0.]).to(self.config['device']))

            image_boxes = torch.cat(image_boxes, dim=0)  # [n_detected_object_for_this_image, 4]
            image_labels = torch.tensor(image_labels)  # [n_detected_object_for_this_image, 1]
            image_scores = torch.cat(image_scores, dim=0)  # [n_detected_object_for_this_image, 1]

            assert len(image_boxes) == len(image_labels) == len(image_scores)
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
        return all_images_boxes, all_images_labels, all_images_scores

class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, config):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.n_p = self.priors_cxcy.size(0)
        self.config = config
        self.priors_xy = utils.cxcy_to_xy(priors_cxcy)
        self.loc_criterion = nn.L1Loss()
        self.cla_criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, loc_output, cla_output, boxes, labels, background_label=0):
        '''
        loc_output: [N, N_P, 4]
        cla_output: [N, N_P, N_CLASSES]
        boxes: list of N tensor, each tensor has size [N_objects, 4], frac. coord
        labels: list of N tensor, each tensor has size [N_objects]

        return loss: scalar
        '''
        loc_gt = torch.zeros_like(loc_output, dtype=torch.float)
        cla_gt = torch.zeros([len(boxes), self.n_p], dtype=torch.long).to(self.config['device'])
        for i in range(len(boxes)):  # for each image in batch
            n_object = len(boxes[i])
            print(boxes[i].shape)
            print(self.priors_xy.shape)
            iou = utils.find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_object, n_p)
            max_overlap_for_each_prior, object_for_each_prior = iou.max(dim=0)  # [n_p], [n_p]

            # make sure all gt boxes corresponds to at least one prior
            _, prior_for_each_object = iou.max(dim=1)  # [n_object]
            object_for_each_prior[prior_for_each_object] = torch.tensor(range(n_object)).to(
                self.config['device'])
            max_overlap_for_each_prior[prior_for_each_object] = 1.

            loc_gt[i] = utils.cxcy_to_gcxgcy(utils.xy_to_cxcy(boxes[i][object_for_each_prior]),
                                             self.priors_cxcy)

            # print(cla_gt.size(), object_for_each_prior.size()), labels[i]
            cla_gt[i] = labels[i][object_for_each_prior]
            # print(cla_gt[i].size(), (max_overlap_for_each_prior < self.config['iou_threshold']).size())
            cla_gt[i][max_overlap_for_each_prior < self.config['iou_threshold']] = utils.class_label[
                'background']

            # get positives

        positives = (cla_gt != background_label)  # [n, n_p]
        n_pos = positives.sum()
        # loc_loss
        self.loc_loss = self.loc_criterion(loc_output[positives], loc_gt[positives])  # scalar
        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # cla_loss, use hard_negative_mining on neg priors
        # print(cla_gt)
        cla_loss = self.cla_criterion(cla_output.view(-1, self.config['n_classes']),
                                      cla_gt.view(-1))  # [N * n_p]
        cla_loss = cla_loss.view(-1, self.n_p)  # [N, n_p], so that we can use tensor positives for

        cla_loss_pos = cla_loss[positives]
        cla_loss_neg = cla_loss[~positives].sort(dim=0, descending=True)[0][
                       :int(n_pos * self.config['pos_neg_ratio'])]
        self.cla_loss = self.config['multiboxloss_loc_cla_ratio'] * (
                    cla_loss_pos.sum() + cla_loss_neg.sum()) / n_pos
        return self.loc_loss + self.cla_loss