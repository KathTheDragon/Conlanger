import pytest
from pytest import raises
from .. import core
from ..core import FormatError, Token, Cat, Word

## Fixtures
@pytest.fixture
def wordtwine():
    return Word('twine')

@pytest.fixture
def categories():
    return {
        'vwl': Cat(['a', 'e', 'i', 'o', 'u'], 'vwl'),
        'cns': Cat(['p', 't', 'k', 's', 'm', 'n', 'r', 'y', 'w'], 'cns')
    }

## core.Token
def test_Token_iter():
    token = Token('TEST', 'test', 0, 0)
    assert list(token) == ['TEST', 'test']

class TestCat:
    def test_Cat(self, categories):
        cat = categories['vwl']
        assert cat.values == ['a', 'e', 'i', 'o', 'u']
        assert cat.name == 'vwl'
        assert str(cat) == '[a, e, i, o, u]'
        assert len(cat) == 5
        assert cat[3] == 'o'
        assert list(cat) == cat.values
        assert 'e' in cat
        assert (cat & ['i', 'y', 'u']).values == ['i', 'u']
        assert (cat + ['y', 'w']).values == ['a', 'e', 'i', 'o', 'u', 'y', 'w']
        assert (cat - ['e', 'o']).values == ['a', 'i', 'u']
        assert cat <= ['a', 'e', 'i', 'o', 'u', 'y', 'w']
        assert ['a', 'e', 'i'] <= cat
        assert not (['a', 'e', 'i', 'o', 'u'] < cat)
        assert cat.index('u') == 4

    def test_Cat_make(self, categories):
        cats = categories  # Shorter alias
        # Nonce categories
        with raises(FormatError):
            Cat.make('a,b')  # No brackets
        with raises(FormatError):
            Cat.make('[a,b')  # Missing right bracket
        with raises(FormatError):
            Cat.make('a,b]')  # Missing left bracket
        with raises(FormatError):
            Cat.make('[a,b,]')  # Trailing comma
        with raises(FormatError):
            Cat.make('[a,,b]')  # Double comma
        with raises(FormatError):
            Cat.make('[a ,b]')  # Invalid space
        with raises(FormatError):
            Cat.make('[a,  b]')  # Invalid space
        with raises(FormatError):
            Cat.make('[a,[b]')  # Invalid left bracket
        with raises(FormatError):
            Cat.make('[a],b]')  # Invalid right bracket
        assert Cat.make('[a,]').values == ['a']  # Trailing comma valid with one item
        assert Cat.make('[a,b]').values == ['a','b']
        assert Cat.make('[a, b]').values == ['a','b']  # Single space valid after comma
        assert Cat.make('[[vwl],y]', cats).values == ['a','e','i','o','u','y']  # Nested categories
        assert Cat.make('[a,b]', name='test').name == 'test'
        # Named categories
        with pytest.raises(FormatError):
            Cat.make('[vwl]')  # No categories given
        with raises(FormatError):
            Cat.make('[stop]', cats)  # Undefined category
        with raises(FormatError):
            Cat.make('[]')  # No name given
        assert Cat.make('[vwl]', cats) is cats['vwl']

class TestWord:
    def test_Word(self, wordtwine):
        from .._pattern import parsePattern
        from ..sce import GlobalEnvironment
        globalenv = GlobalEnvironment(parsePattern('e#'), None)
        globalenvindices = GlobalEnvironment(parsePattern('e#'), [4,5])
        assert wordtwine.phones == ['#','t','w','i','n','e','#']
        assert Word(['t','w','i','n','e']).phones == ['t','w','i','n','e']
        assert repr(wordtwine) == "Word('twine')"
        assert str(wordtwine) == 'twine'
        assert len(wordtwine) == 7
        assert wordtwine[3] == 'i'
        assert wordtwine[3:5].phones == ['i','n']
        assert list(wordtwine) == wordtwine.phones
        assert Word(['w','i']) in wordtwine
        assert globalenv in wordtwine
        assert globalenvindices in wordtwine
        assert 'w' in wordtwine
        assert ('the' + wordtwine).phones == ['t','h','e','#','t','w','i','n','e','#']
        assert (Word('the') + wordtwine).phones == ['#','t','h','e','#','t','w','i','n','e','#']
        assert (wordtwine * 2).phones == ['#','t','w','i','n','e','#','t','w','i','n','e','#']
        assert wordtwine.strip().phones == ['t','w','i','n','e']

    def test_Word_find(self, wordtwine, categories):
        from .._pattern import parsePattern
        cats = categories
        comparison = parsePattern('[cns]{=3}', cats)
        assert wordtwine.find(comparison) == 1
        assert wordtwine.find(comparison, start=2) == -1
        assert wordtwine.find(comparison, end=4) == -1
        pattern = parsePattern('[cns][vwl]', cats)
        assert wordtwine.find(pattern) == 2
        assert wordtwine.find(pattern, start=3) == 4
        assert wordtwine.find(pattern, end=3) == -1

    @pytest.mark.skip(reason='Not really anything to test at this location')
    def test_Word_matchPattern(self):
        pass

    def test_Word_matchEnv(self, wordtwine):
        from .._pattern import parsePattern
        from ..sce import GlobalEnvironment, LocalEnvironment
        environment = [
            None,
            GlobalEnvironment(parsePattern('e#'), None),
            LocalEnvironment(parsePattern('w'), parsePattern('n'))
        ]
        assert wordtwine.matchEnv(environment, 3, 4)

    def test_Word_applyMatch(self):
        pass

class TestSyllabifier:
    pass

class TestFunctions:
    def test_resolveTargetRef(self):
        pass

    def test_sliceIndices(self):
        pass

    def test_parseCats(self):
        pass

    def test_parseWord(self):
        pass

    def test_unparseWord(self):
        pass

    def test_split(self):
        pass

    def test_partition(self):
        pass

    def test_partitionTokens(self):
        pass
