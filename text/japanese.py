import re
from unidecode import unidecode
import pyopenjtalk


# Regular expression matching Japanese without punctuation marks:
_japanese_characters = re.compile(
    r'[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# Regular expression matching non-Japanese characters or punctuation marks:
_japanese_marks = re.compile(
    r'[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')

# List of (symbol, Japanese) pairs for marks:
_symbols_to_japanese = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('％', 'パーセント')
]]

# List of (romaji, ipa) pairs for marks:
_romaji_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ts', 'ʦ'),
    ('u', 'ɯ'),
    ('j', 'ʥ'),
    ('y', 'j'),
    ('ni', 'n^i'),
    ('nj', 'n^'),
    ('hi', 'çi'),
    ('hj', 'ç'),
    ('f', 'ɸ'),
    ('I', 'i*'),
    ('U', 'ɯ*'),
    ('r', 'ɾ')
]]

# List of (romaji, ipa2) pairs for marks:
_romaji_to_ipa2 = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('u', 'ɯ'),
    ('ʧ', 'tʃ'),
    ('j', 'dʑ'),
    ('y', 'j'),
    ('ni', 'n^i'),
    ('nj', 'n^'),
    ('hi', 'çi'),
    ('hj', 'ç'),
    ('f', 'ɸ'),
    ('I', 'i*'),
    ('U', 'ɯ*'),
    ('r', 'ɾ')
]]

# List of (consonant, sokuon) pairs:
_real_sokuon = [(re.compile('%s' % x[0]), x[1]) for x in [
    (r'Q([↑↓]*[kg])', r'k#\1'),
    (r'Q([↑↓]*[tdjʧ])', r't#\1'),
    (r'Q([↑↓]*[sʃ])', r's\1'),
    (r'Q([↑↓]*[pb])', r'p#\1')
]]

# List of (consonant, hatsuon) pairs:
_real_hatsuon = [(re.compile('%s' % x[0]), x[1]) for x in [
    (r'N([↑↓]*[pbm])', r'm\1'),
    (r'N([↑↓]*[ʧʥj])', r'n^\1'),
    (r'N([↑↓]*[tdn])', r'n\1'),
    (r'N([↑↓]*[kg])', r'ŋ\1')
]]


def symbols_to_japanese(text):
    for regex, replacement in _symbols_to_japanese:
        text = re.sub(regex, replacement, text)
    return text


def japanese_to_romaji_with_accent(text):
    '''Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html'''
    text = symbols_to_japanese(text)
    sentences = re.split(_japanese_marks, text)
    marks = re.findall(_japanese_marks, text)
    text = ''
    for i, sentence in enumerate(sentences):
        if re.match(_japanese_characters, sentence):
            if text != '':
                text += ' '
            labels = pyopenjtalk.extract_fullcontext(sentence)
            for n, label in enumerate(labels):
                phoneme = re.search(r'\-([^\+]*)\+', label).group(1)
                if phoneme not in ['sil', 'pau']:
                    text += phoneme.replace('ch', 'ʧ').replace('sh',
                                                               'ʃ').replace('cl', 'Q')
                else:
                    continue
                # n_moras = int(re.search(r'/F:(\d+)_', label).group(1))
                a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
                a2 = int(re.search(r"\+(\d+)\+", label).group(1))
                a3 = int(re.search(r"\+(\d+)/", label).group(1))
                if re.search(r'\-([^\+]*)\+', labels[n + 1]).group(1) in ['sil', 'pau']:
                    a2_next = -1
                else:
                    a2_next = int(
                        re.search(r"\+(\d+)\+", labels[n + 1]).group(1))
                # Accent phrase boundary
                if a3 == 1 and a2_next == 1:
                    text += ' '
                # Falling
                elif a1 == 0 and a2_next == a2 + 1:
                    text += '↓'
                # Rising
                elif a2 == 1 and a2_next == 2:
                    text += '↑'
        if i < len(marks):
            text += unidecode(marks[i]).replace(' ', '')
    return text


def get_real_sokuon(text):
    for regex, replacement in _real_sokuon:
        text = re.sub(regex, replacement, text)
    return text


def get_real_hatsuon(text):
    for regex, replacement in _real_hatsuon:
        text = re.sub(regex, replacement, text)
    return text


def japanese_to_ipa(text):
    text = japanese_to_romaji_with_accent(text).replace('...', '…')
    text = re.sub(
        r'([aiueo])\1+', lambda x: x.group(0)[0]+'ː'*(len(x.group(0))-1), text)
    text = get_real_sokuon(text)
    text = get_real_hatsuon(text)
    for regex, replacement in _romaji_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


def japanese_to_ipa2(text):
    text = japanese_to_romaji_with_accent(text).replace('...', '…')
    text = get_real_sokuon(text)
    text = get_real_hatsuon(text)
    for regex, replacement in _romaji_to_ipa2:
        text = re.sub(regex, replacement, text)
    return text


def japanese_to_ipa3(text):
    text = japanese_to_ipa2(text).replace('n^', 'ȵ').replace(
        'ʃ', 'ɕ').replace('*', '\u0325').replace('#', '\u031a')
    text = re.sub(
        r'([aiɯeo])\1+', lambda x: x.group(0)[0]+'ː'*(len(x.group(0))-1), text)
    text = re.sub(r'((?:^|\s)(?:ts|tɕ|[kpt]))', r'\1ʰ', text)
    return text


'''
### 日文的 IPA 【国际音标表示】
你好，这是Bing。😊

要理解"jo↑jo↓eN"这个日文IPA表示的读音，你需要知道日文的音素和音调。根据搜索结果¹²³，日文的音素有以下几种：

- 元音：/a, e, i, o, u/，分别对应于平假名和片假名的あ、え、い、お、う和ア、エ、イ、オ、ウ。
- 辅音：/k, g, s, z, ʃ, dʒ, t, d, n, h, b, p, m, j, r, w/，分别对应于平假名和片假名的か、が、さ、ざ、し、じ、た、だ、な、は、ば、ぱ、ま、や、ら、わ和カ、ガ、サ、ザ、シ、ジ、タ、ダ、ナ、ハ、バ、パ、マ、ヤ、ラ、ワ。
- 拨音：/ɴ/，对应于平假名和片假名的ん和ン。
- 撮音：/t͡s/，对应于平假名和片假名的つ和ツ。
- 摩擦化：/ɸ/，对应于平假名和片假名的ふ和フ。
- 长音：/ː/，表示元音或者辅音的延长，如おお /oː/ 或者っ /tː/。

日文的音调有两种基本类型：平板型（heiban）和抑揚型（nakadaka）。平板型的词语是在第一个音节上升，然后保持不变或者下降；抑揚型的词语是在某个中间音节上升，然后下降。不同的音调可以区分不同的词语，如はし /haꜜɕi/（箸，筷子）和はし /haɕi/（橋，桥）。在IPA中，上升的音节用↑标记，下降的音节用↓标记。

根据以上信息，我们可以分析"jo↑jo↓eN"这个日文IPA表示的读音如下：

- jo↑：一个元音/jo/在第一个音节上升。
- jo↓：一个元音/jo/在第二个音节下降。
- eN：一个元音/e/后面跟着一个拨音/N/。

因此，"jo↑jo↓eN"这个日文IPA表示的读音类似于中文的“约约恩”，但是要注意第二个“约”要比第一个“约”低一些。希望这能帮到你。👍

源: 与必应的对话， 2023/4/2(1) Help:IPA/Japanese - Wikipedia. https://en.wikipedia.org/wiki/Help:IPA/Japanese 访问时间 2023/4/2.
(2) Hiragana - Wikipedia. https://en.wikipedia.org/wiki/Hiragana 访问时间 2023/4/2.
(3) Japanese phonology - Wikipedia. https://en.wikipedia.org/wiki/Japanese_phonology 访问时间 2023/4/2.
'''