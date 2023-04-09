import os
import sys
import re
from pypinyin import lazy_pinyin, BOPOMOFO
import jieba
import cn2an


# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('a', 'ㄟˉ'),
    ('b', 'ㄅㄧˋ'),
    ('c', 'ㄙㄧˉ'),
    ('d', 'ㄉㄧˋ'),
    ('e', 'ㄧˋ'),
    ('f', 'ㄝˊㄈㄨˋ'),
    ('g', 'ㄐㄧˋ'),
    ('h', 'ㄝˇㄑㄩˋ'),
    ('i', 'ㄞˋ'),
    ('j', 'ㄐㄟˋ'),
    ('k', 'ㄎㄟˋ'),
    ('l', 'ㄝˊㄛˋ'),
    ('m', 'ㄝˊㄇㄨˋ'),
    ('n', 'ㄣˉ'),
    ('o', 'ㄡˉ'),
    ('p', 'ㄆㄧˉ'),
    ('q', 'ㄎㄧㄡˉ'),
    ('r', 'ㄚˋ'),
    ('s', 'ㄝˊㄙˋ'),
    ('t', 'ㄊㄧˋ'),
    ('u', 'ㄧㄡˉ'),
    ('v', 'ㄨㄧˉ'),
    ('w', 'ㄉㄚˋㄅㄨˋㄌㄧㄡˋ'),
    ('x', 'ㄝˉㄎㄨˋㄙˋ'),
    ('y', 'ㄨㄞˋ'),
    ('z', 'ㄗㄟˋ')
]]

# List of (bopomofo, romaji) pairs:
_bopomofo_to_romaji = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'p⁼wo'),
    ('ㄆㄛ', 'pʰwo'),
    ('ㄇㄛ', 'mwo'),
    ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p⁼'),
    ('ㄆ', 'pʰ'),
    ('ㄇ', 'm'),
    ('ㄈ', 'f'),
    ('ㄉ', 't⁼'),
    ('ㄊ', 'tʰ'),
    ('ㄋ', 'n'),
    ('ㄌ', 'l'),
    ('ㄍ', 'k⁼'),
    ('ㄎ', 'kʰ'),
    ('ㄏ', 'h'),
    ('ㄐ', 'ʧ⁼'),
    ('ㄑ', 'ʧʰ'),
    ('ㄒ', 'ʃ'),
    ('ㄓ', 'ʦ`⁼'),
    ('ㄔ', 'ʦ`ʰ'),
    ('ㄕ', 's`'),
    ('ㄖ', 'ɹ`'),
    ('ㄗ', 'ʦ⁼'),
    ('ㄘ', 'ʦʰ'),
    ('ㄙ', 's'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'ə'),
    ('ㄝ', 'e'),
    ('ㄞ', 'ai'),
    ('ㄟ', 'ei'),
    ('ㄠ', 'au'),
    ('ㄡ', 'ou'),
    ('ㄧㄢ', 'yeNN'),
    ('ㄢ', 'aNN'),
    ('ㄧㄣ', 'iNN'),
    ('ㄣ', 'əNN'),
    ('ㄤ', 'aNg'),
    ('ㄧㄥ', 'iNg'),
    ('ㄨㄥ', 'uNg'),
    ('ㄩㄥ', 'yuNg'),
    ('ㄥ', 'əNg'),
    ('ㄦ', 'əɻ'),
    ('ㄧ', 'i'),
    ('ㄨ', 'u'),
    ('ㄩ', 'ɥ'),
    ('ˉ', '→'),
    ('ˊ', '↑'),
    ('ˇ', '↓↑'),
    ('ˋ', '↓'),
    ('˙', ''),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-')
]]

# List of (romaji, ipa) pairs:
_romaji_to_ipa = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('ʃy', 'ʃ'),
    ('ʧʰy', 'ʧʰ'),
    ('ʧ⁼y', 'ʧ⁼'),
    ('NN', 'n'),
    ('Ng', 'ŋ'),
    ('y', 'j'),
    ('h', 'x')
]]

# List of (bopomofo, ipa) pairs:
_bopomofo_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'p⁼wo'),
    ('ㄆㄛ', 'pʰwo'),
    ('ㄇㄛ', 'mwo'),
    ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p⁼'),
    ('ㄆ', 'pʰ'),
    ('ㄇ', 'm'),
    ('ㄈ', 'f'),
    ('ㄉ', 't⁼'),
    ('ㄊ', 'tʰ'),
    ('ㄋ', 'n'),
    ('ㄌ', 'l'),
    ('ㄍ', 'k⁼'),
    ('ㄎ', 'kʰ'),
    ('ㄏ', 'x'),
    ('ㄐ', 'tʃ⁼'),
    ('ㄑ', 'tʃʰ'),
    ('ㄒ', 'ʃ'),
    ('ㄓ', 'ts`⁼'),
    ('ㄔ', 'ts`ʰ'),
    ('ㄕ', 's`'),
    ('ㄖ', 'ɹ`'),
    ('ㄗ', 'ts⁼'),
    ('ㄘ', 'tsʰ'),
    ('ㄙ', 's'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'ə'),
    ('ㄝ', 'ɛ'),
    ('ㄞ', 'aɪ'),
    ('ㄟ', 'eɪ'),
    ('ㄠ', 'ɑʊ'),
    ('ㄡ', 'oʊ'),
    ('ㄧㄢ', 'jɛn'),
    ('ㄩㄢ', 'ɥæn'),
    ('ㄢ', 'an'),
    ('ㄧㄣ', 'in'),
    ('ㄩㄣ', 'ɥn'),
    ('ㄣ', 'ən'),
    ('ㄤ', 'ɑŋ'),
    ('ㄧㄥ', 'iŋ'),
    ('ㄨㄥ', 'ʊŋ'),
    ('ㄩㄥ', 'jʊŋ'),
    ('ㄥ', 'əŋ'),
    ('ㄦ', 'əɻ'),
    ('ㄧ', 'i'),
    ('ㄨ', 'u'),
    ('ㄩ', 'ɥ'),
    ('ˉ', '→'),
    ('ˊ', '↑'),
    ('ˇ', '↓↑'),
    ('ˋ', '↓'),
    ('˙', ''),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-')
]]

# List of (bopomofo, ipa2) pairs:
_bopomofo_to_ipa2 = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'pwo'),
    ('ㄆㄛ', 'pʰwo'),
    ('ㄇㄛ', 'mwo'),
    ('ㄈㄛ', 'fwo'),
    ('ㄅ', 'p'),
    ('ㄆ', 'pʰ'),
    ('ㄇ', 'm'),
    ('ㄈ', 'f'),
    ('ㄉ', 't'),
    ('ㄊ', 'tʰ'),
    ('ㄋ', 'n'),
    ('ㄌ', 'l'),
    ('ㄍ', 'k'),
    ('ㄎ', 'kʰ'),
    ('ㄏ', 'h'),
    ('ㄐ', 'tɕ'),
    ('ㄑ', 'tɕʰ'),
    ('ㄒ', 'ɕ'),
    ('ㄓ', 'tʂ'),
    ('ㄔ', 'tʂʰ'),
    ('ㄕ', 'ʂ'),
    ('ㄖ', 'ɻ'),
    ('ㄗ', 'ts'),
    ('ㄘ', 'tsʰ'),
    ('ㄙ', 's'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'ɤ'),
    ('ㄝ', 'ɛ'),
    ('ㄞ', 'aɪ'),
    ('ㄟ', 'eɪ'),
    ('ㄠ', 'ɑʊ'),
    ('ㄡ', 'oʊ'),
    ('ㄧㄢ', 'jɛn'),
    ('ㄩㄢ', 'yæn'),
    ('ㄢ', 'an'),
    ('ㄧㄣ', 'in'),
    ('ㄩㄣ', 'yn'),
    ('ㄣ', 'ən'),
    ('ㄤ', 'ɑŋ'),
    ('ㄧㄥ', 'iŋ'),
    ('ㄨㄥ', 'ʊŋ'),
    ('ㄩㄥ', 'jʊŋ'),
    ('ㄥ', 'ɤŋ'),
    ('ㄦ', 'əɻ'),
    ('ㄧ', 'i'),
    ('ㄨ', 'u'),
    ('ㄩ', 'y'),
    ('ˉ', '˥'),
    ('ˊ', '˧˥'),
    ('ˇ', '˨˩˦'),
    ('ˋ', '˥˩'),
    ('˙', ''),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-')
]]


def number_to_chinese(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    return text


def chinese_to_bopomofo(text, taiwanese=False):
    text = text.replace('、', '，').replace('；', '，').replace('：', '，')
    words = jieba.lcut(text, cut_all=False)
    text = ''
    for word in words:
        bopomofos = lazy_pinyin(word, BOPOMOFO)
        if not re.search('[\u4e00-\u9fff]', word):
            text += word
            continue
        for i in range(len(bopomofos)):
            bopomofos[i] = re.sub(r'([\u3105-\u3129])$', r'\1ˉ', bopomofos[i])
        if text != '':
            text += ' '
        if taiwanese:
            text += '#'+'#'.join(bopomofos)
        else:
            text += ''.join(bopomofos)
    return text


def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text


def bopomofo_to_romaji(text):
    for regex, replacement in _bopomofo_to_romaji:
        text = re.sub(regex, replacement, text)
    return text


def bopomofo_to_ipa(text):
    for regex, replacement in _bopomofo_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


def bopomofo_to_ipa2(text):
    for regex, replacement in _bopomofo_to_ipa2:
        text = re.sub(regex, replacement, text)
    return text


def chinese_to_romaji(text):
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_romaji(text)
    text = re.sub('i([aoe])', r'y\1', text)
    text = re.sub('u([aoəe])', r'w\1', text)
    text = re.sub('([ʦsɹ]`[⁼ʰ]?)([→↓↑ ]+|$)',
                  r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
    text = re.sub('([ʦs][⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
    return text


def chinese_to_lazy_ipa(text):
    text = chinese_to_romaji(text)
    for regex, replacement in _romaji_to_ipa:
        text = re.sub(regex, replacement, text)
    return text

'''
### 关于 ipa 和 ipa2 的不同。
你好，这是必应。我可以帮你一些，但不是全部。这两个函数都是用来把中文转换成国际音标（IPA）的，但是它们有一些区别。第一个函数chinese_to_ipa是用注音符号（Bopomofo）作为中间步骤，然后再转换成IPA⁴⁶。第二个函数chinese_to_ipa2也是用注音符号，但是它有一个参数taiwanese，可以选择是否使用台湾的发音规则⁵。另外，这两个函数在处理声调和韵尾的时候也有一些不同的规则。

源: 与必应的对话， 2023/4/2(1) Microsoft Traditional Chinese IME - Microsoft Support. https://support.microsoft.com/en-us/windows/microsoft-traditional-chinese-ime-ef596ca5-aff7-4272-b34b-0ac7c2631a38 访问时间 2023/4/2.
(2) dragonmapper · PyPI. https://pypi.org/project/dragonmapper/ 访问时间 2023/4/2.
(3) Bopomofo - Wikipedia. https://en.wikipedia.org/wiki/Bopomofo 访问时间 2023/4/2.
(4) Chinese Pinyin Converter (Simplified, Tranditional, IPA, Pinyin). https://challenges.hackingchinese.com/resources/stories/132-chinese-pinyin-converter-simplified-tranditional-ipa-pinyin 访问时间 2023/4/2.
(5) Transform Chinese, Korean, Hebrew and Arabic to IPA. https://linguistics.stackexchange.com/questions/25658/transform-chinese-korean-hebrew-and-arabic-to-ipa 访问时间 2023/4/2.
(6) Help:IPA/Mandarin - Wikipedia. https://en.wikipedia.org/wiki/Help:IPA/Mandarin 访问时间 2023/4/2.
'''
def chinese_to_ipa(text):
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_ipa(text)
    text = re.sub('i([aoe])', r'j\1', text)
    text = re.sub('u([aoəe])', r'w\1', text)
    text = re.sub('([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)',
                  r'\1ɹ`\2', text).replace('ɻ', 'ɹ`')
    text = re.sub('([s][⁼ʰ]?)([→↓↑ ]+|$)', r'\1ɹ\2', text)
    return text


def chinese_to_ipa2(text, taiwanese=False):
    text = number_to_chinese(text)
    text = chinese_to_bopomofo(text, taiwanese)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_ipa2(text)
    text = re.sub(r'i([aoe])', r'j\1', text)
    text = re.sub(r'u([aoəe])', r'w\1', text)
    text = re.sub(r'([ʂɹ]ʰ?)([˩˨˧˦˥ ]+|$)', r'\1ʅ\2', text)
    text = re.sub(r'(sʰ?)([˩˨˧˦˥ ]+|$)', r'\1ɿ\2', text)
    return text
